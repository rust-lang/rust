//@ edition: 2021
//@ run-pass
// ignore-tidy-linelength
//@ run-flags: --sysroot {{sysroot-base}} {{src-base}}/auxiliary/dependent-evidence-codec-input.rs
//@ ignore-cross-compile
//@ ignore-remote
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)

#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_serialize;
extern crate rustc_span;
extern crate rustc_type_ir;

use std::cell::Cell;
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::ExitCode;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hash::{
    RawDefId, RawSpan, StableHash, StableHashControls, StableHashCtxt, StableHasher,
};
use rustc_driver::{Callbacks, Compilation};
use rustc_errors::DiagCtxt;
use rustc_errors::emitter::SilentEmitter;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex};
use rustc_interface::interface::Compiler;
use rustc_middle::ich::StableHashState;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::traits::solve::{
    BoundRequiredContract, CandidateEvidence, CandidateEvidenceSource, CandidateEvidenceUse,
    EvidenceProjection, TraitEvidence, TraitEvidenceData, TraitEvidenceKind,
};
use rustc_middle::ty::codec::{SHORTHAND_OFFSET, TyDecoder, TyEncoder};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, Upcast};
use rustc_serialize::opaque::mem_encoder::MemEncoder;
use rustc_serialize::opaque::{MAGIC_END_BYTES, MemDecoder};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::{
    BlobDecoder, ByteSymbol, ExpnId, Span, SpanDecoder, SpanEncoder, Symbol, SyntaxContext,
};

fn main() -> ExitCode {
    rustc_driver::catch_with_exit_code(|| {
        rustc_driver::run_compiler(&std::env::args().collect::<Vec<_>>(), &mut Check);
    })
}

struct Check;

impl Callbacks for Check {
    fn after_analysis<'tcx>(&mut self, _: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        tcx.dcx().abort_if_errors();
        let family = find_item(tcx, "Family");
        let item = associated_item(tcx, family);
        let other_item = associated_item(tcx, find_item(tcx, "Other"));
        let mut impls = tcx.all_impls(family);
        let first = impls.next().unwrap();
        let second = impls.next().unwrap();
        let (leaf_impl, pair_impl) =
            if tcx.type_of(first).instantiate_identity().skip_norm_wip() == tcx.types.bool {
                (first, second)
            } else {
                (second, first)
            };
        let trait_ref = ty::TraitRef::new(tcx, family, [tcx.types.bool]);
        let leaf = tcx.mk_trait_evidence(CandidateEvidence::new(
            trait_ref,
            CandidateEvidenceSource::Impl { impl_def_id: leaf_impl, args: tcx.mk_args(&[]) },
            [],
        ));
        test_states(tcx, item, leaf);
        test_shared_dag(tcx, item, pair_impl, leaf);
        test_decode_validation(tcx, other_item, leaf);
        Compilation::Stop
    }
}

fn find_item(tcx: TyCtxt<'_>, name: &str) -> DefId {
    tcx.hir_crate_items(())
        .free_items()
        .map(|item| item.owner_id.to_def_id())
        .find(|&id| tcx.opt_item_name(id).is_some_and(|item| item.as_str() == name))
        .unwrap()
}

fn associated_item(tcx: TyCtxt<'_>, trait_id: DefId) -> DefId {
    tcx.associated_items(trait_id).in_definition_order().next().unwrap().def_id
}

fn projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: DefId,
    evidence: TraitEvidence<'tcx>,
) -> EvidenceProjection<'tcx> {
    tcx.mk_evidence_projection(ty::EvidenceProjectionData { item_def_id: item, evidence })
}

fn test_states<'tcx>(tcx: TyCtxt<'tcx>, item: DefId, selected: TraitEvidence<'tcx>) {
    round_trip(tcx, selected);
    let trait_ref = selected.trait_ref;
    let bound = tcx.mk_trait_evidence_kind(
        trait_ref,
        TraitEvidenceKind::Bound(
            ty::BoundVarIndexKind::Bound(ty::DebruijnIndex::from_u32(1)),
            ty::BoundEvidence::new(ty::BoundVar::from_u32(0)),
        ),
    );
    let clause = ty::ClauseKind::Trait(ty::TraitClause {
        trait_ref,
        polarity: ty::ClausePolarity::Positive,
    });
    let contract = tcx.mk_bound_required_contract(ty::BoundRequiredContractData {
        identity: ty::solve::InstantiatedItemContract {
            key: ty::solve::ItemContractKey { owner: trait_ref.def_id, hir_local_id: 0 },
            complete_early_args: tcx.mk_args(&[]),
        },
        clauses: tcx.mk_clauses(&[ty::Binder::dummy(clause).upcast(tcx)]),
        principal_index: 0,
        ordinary_args: None,
    });
    let vars =
        tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Evidence(ty::EvidenceVariable {
            clause,
            required_contract: Some(contract),
        })]);
    let inner =
        ty::Binder::bind_with_vars(projection(tcx, item, bound), tcx.mk_bound_variable_kinds(&[]));
    let outer = ty::Binder::bind_with_vars(inner, vars);
    assert!(inner.has_escaping_bound_vars());
    assert!(!outer.has_escaping_bound_vars());
    // Fragment decoding preserves a reference to an outer binder until it is decoded too.
    round_trip(tcx, (outer, outer));
    round_trip(tcx, contract);

    let placeholder = tcx.mk_trait_evidence_kind(
        trait_ref,
        TraitEvidenceKind::Placeholder(ty::PlaceholderEvidence::new_anon(
            ty::UniverseIndex::from_u32(3),
            ty::BoundVar::from_u32(2),
        )),
    );
    assert!(placeholder.has_placeholders());
    assert!(!placeholder.has_escaping_bound_vars());
    round_trip(tcx, projection(tcx, item, placeholder));

    // Error evidence carries a real emission guarantee, and follows its codec rejection.
    let dcx = DiagCtxt::new(Box::new(SilentEmitter));
    let guar = dcx.handle().err("error evidence serialization test");
    let error = tcx.mk_trait_evidence_kind(trait_ref, TraitEvidenceKind::Error(guar));
    assert!(error.references_error());
    let mut encoder = TestEncoder::new();
    assert_panics("should never serialize an `ErrorGuaranteed`", || error.encode(&mut encoder));
    let bytes = encoder.finish();
    assert_panics("`ErrorGuaranteed` should never have been serialized", || {
        let _: TraitEvidence<'_> = Decodable::decode(&mut TestDecoder::new(tcx, &bytes));
    });
}

fn test_shared_dag<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: DefId,
    pair_impl: DefId,
    leaf: TraitEvidence<'tcx>,
) {
    let mut evidence = leaf;
    let mut halfway_size = 0;
    const DEPTH: usize = 32;
    for depth in 1..=DEPTH {
        let child_ty = evidence.trait_ref.self_ty();
        let self_ty = Ty::new_tup(tcx, &[child_ty, child_ty]);
        evidence = tcx.mk_trait_evidence(CandidateEvidence::new(
            ty::TraitRef::new(tcx, leaf.trait_ref.def_id, [self_ty]),
            CandidateEvidenceSource::Impl {
                impl_def_id: pair_impl,
                args: tcx.mk_args(&[child_ty.into()]),
            },
            [CandidateEvidenceUse::Instantiated(evidence); 2],
        ));
        if depth == DEPTH / 2 {
            let mut encoder = TestEncoder::new();
            evidence.encode(&mut encoder);
            halfway_size = encoder.position();
        }
    }

    let projected = projection(tcx, item, evidence);
    let alias = ty::AliasTy::new_from_args(
        tcx,
        ty::EvidenceProjection { projection: projected },
        tcx.mk_args(&[]),
    );
    // Computing cached flags must not expand the 33 proof handles into 2^32 paths.
    let projected_ty = Ty::new_alias(tcx, ty::IsRigid::No, alias);
    assert!(projected_ty.has_evidence_projections());
    assert!(!projected_ty.has_escaping_bound_vars());

    let hash = tcx.with_stable_hashing_context(|inner| {
        let mut hcx = CountingHashCtxt { inner, def_ids: Cell::new(0) };
        let mut hasher = StableHasher::new();
        evidence.stable_hash(&mut hcx, &mut hasher);
        assert!(hcx.def_ids.get() <= 10 * (DEPTH + 1));
        hasher.finish::<Fingerprint>()
    });
    assert_ne!(hash, fingerprint(tcx, leaf));

    let mut encoder = TestEncoder::new();
    evidence.encode(&mut encoder);
    let size = encoder.position();
    assert!(size <= 3 * halfway_size);
    assert_eq!(encoder.evidence.len(), DEPTH + 1);
    evidence.encode(&mut encoder);
    assert!(encoder.position() - size <= 10);
    let bytes = encoder.finish();
    let mut decoder = TestDecoder::new(tcx, &bytes);
    let decoded: TraitEvidence<'_> = Decodable::decode(&mut decoder);
    assert_eq!(decoded, evidence);
    assert_eq!(decoder.evidence.len(), DEPTH + 1);
    assert_eq!(fingerprint(tcx, decoded), hash);
    assert_eq!(TraitEvidence::decode(&mut decoder), evidence);
    assert_eq!(decoder.position(), bytes.len() - MAGIC_END_BYTES.len());
    let mut current = decoded;
    for _ in 0..DEPTH {
        let TraitEvidenceKind::Selected(recipe) = &current.kind else { panic!() };
        let children = &recipe.root_node().nested_evidence;
        assert_eq!(children[0], children[1]);
        let CandidateEvidenceUse::Instantiated(child) = children[0];
        current = child;
    }
    assert_eq!(current, leaf);
}

fn test_decode_validation<'tcx>(
    tcx: TyCtxt<'tcx>,
    other_item: DefId,
    evidence: TraitEvidence<'tcx>,
) {
    let TraitEvidenceKind::Selected(recipe) = &evidence.kind else { panic!() };
    for (edge, expected) in [
        (1, "nested trait proof node is out of bounds"),
        (0, "non-productive cycle in trait proof DAG"),
    ] {
        let mut recipe = recipe.clone();
        recipe.nodes[0].nested.push(edge);
        let data = TraitEvidenceData {
            trait_ref: evidence.trait_ref,
            kind: TraitEvidenceKind::Selected(recipe),
        };
        let mut encoder = TestEncoder::new();
        encoder.emit_usize(0);
        data.encode(&mut encoder);
        let bytes = encoder.finish();
        assert_panics(expected, || {
            let _: TraitEvidence<'_> = Decodable::decode(&mut TestDecoder::new(tcx, &bytes));
        });
    }

    let mut encoder = TestEncoder::new();
    ty::EvidenceProjectionData::<TyCtxt<'tcx>> { item_def_id: other_item, evidence }
        .encode(&mut encoder);
    let bytes = encoder.finish();
    assert_panics("is not owned by evidence trait", || {
        let _: EvidenceProjection<'_> = Decodable::decode(&mut TestDecoder::new(tcx, &bytes));
    });

    let mut encoder = TestEncoder::new();
    encoder.emit_usize(SHORTHAND_OFFSET);
    let bytes = encoder.finish();
    assert_panics("trait evidence shorthand must refer to earlier data", || {
        let _: TraitEvidence<'_> = Decodable::decode(&mut TestDecoder::new(tcx, &bytes));
    });

    // A backward reference can still be cyclic if its inline ancestor is unfinished.
    let parent = tcx.mk_trait_evidence(CandidateEvidence::new(
        evidence.trait_ref,
        recipe.root_source(),
        [CandidateEvidenceUse::Instantiated(evidence)],
    ));
    let mut encoder = TestEncoder::new();
    parent.encode(&mut encoder);
    let child_position = encoder.evidence[&evidence] - SHORTHAND_OFFSET;
    encoder.opaque.data.truncate(child_position);
    encoder.emit_usize(SHORTHAND_OFFSET);
    encoder.types.clear();
    encoder.predicates.clear();
    encoder.evidence.clear();
    let valid_position = encoder.position();
    evidence.encode(&mut encoder);
    let bytes = encoder.finish();
    let mut decoder = TestDecoder::new(tcx, &bytes);
    assert_panics("cycle in trait evidence shorthands", || {
        let _: TraitEvidence<'_> = Decodable::decode(&mut decoder);
    });
    assert!(decoder.in_progress.is_empty());
    assert_eq!(decoder.with_position(valid_position, TraitEvidence::decode), evidence);

    let invalid_contract: ty::BoundRequiredContractData<TyCtxt<'tcx>> =
        ty::BoundRequiredContractData {
            identity: ty::solve::InstantiatedItemContract {
                key: ty::solve::ItemContractKey {
                    owner: evidence.trait_ref.def_id,
                    hir_local_id: 0,
                },
                complete_early_args: tcx.mk_args(&[]),
            },
            clauses: tcx.mk_clauses(&[]),
            principal_index: 0,
            ordinary_args: None,
        };
    let mut encoder = TestEncoder::new();
    invalid_contract.encode(&mut encoder);
    let bytes = encoder.finish();
    assert_panics("required-contract principal must be a trait clause", || {
        let _: BoundRequiredContract<'_> = Decodable::decode(&mut TestDecoder::new(tcx, &bytes));
    });
}

fn round_trip<'tcx, T>(tcx: TyCtxt<'tcx>, value: T)
where
    T: Debug + PartialEq + StableHash + Encodable<TestEncoder<'tcx>>,
    T: for<'a> Decodable<TestDecoder<'a, 'tcx>>,
{
    let mut encoder = TestEncoder::new();
    value.encode(&mut encoder);
    let bytes = encoder.finish();
    let mut decoder = TestDecoder::new(tcx, &bytes);
    let decoded: T = Decodable::decode(&mut decoder);
    assert_eq!(value, decoded);
    assert_eq!(fingerprint(tcx, &value), fingerprint(tcx, &decoded));
    assert_eq!(decoder.position(), bytes.len() - MAGIC_END_BYTES.len());
}

fn fingerprint(tcx: TyCtxt<'_>, value: impl StableHash) -> Fingerprint {
    tcx.with_stable_hashing_context(|mut hcx| {
        let mut hasher = StableHasher::new();
        value.stable_hash(&mut hcx, &mut hasher);
        hasher.finish()
    })
}

struct CountingHashCtxt<'a> {
    inner: StableHashState<'a>,
    def_ids: Cell<usize>,
}

impl StableHashCtxt for CountingHashCtxt<'_> {
    fn stable_hash_span(&mut self, span: RawSpan, hasher: &mut StableHasher) {
        self.inner.stable_hash_span(span, hasher);
    }
    fn def_path_hash(&self, def_id: RawDefId) -> Fingerprint {
        self.def_ids.set(self.def_ids.get() + 1);
        self.inner.def_path_hash(def_id)
    }
    fn stable_hash_controls(&self) -> StableHashControls {
        self.inner.stable_hash_controls()
    }
    fn assert_default_stable_hash_controls(&self, message: &str) {
        self.inner.assert_default_stable_hash_controls(message);
    }
}

fn assert_panics(expected: &str, f: impl FnOnce()) {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    let error = result.expect_err("invalid representation was accepted");
    let message = error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(message.contains(expected), "unexpected panic: {message}");
}

// These adapters use the production type codecs and a real TyCtxt. DefIds remain
// local to this test's compiler invocation; the opaque codecs handle primitives.
struct TestEncoder<'tcx> {
    opaque: MemEncoder,
    types: FxHashMap<Ty<'tcx>, usize>,
    predicates: FxHashMap<ty::PredicateKind<'tcx>, usize>,
    evidence: FxHashMap<TraitEvidence<'tcx>, usize>,
}

impl<'tcx> TestEncoder<'tcx> {
    fn new() -> Self {
        Self {
            opaque: MemEncoder::new(),
            types: Default::default(),
            predicates: Default::default(),
            evidence: Default::default(),
        }
    }
    fn finish(self) -> Vec<u8> {
        let mut bytes = self.opaque.finish();
        bytes.extend_from_slice(MAGIC_END_BYTES);
        bytes
    }
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, value: $ty) { self.opaque.$name(value); })*
    };
}

impl Encoder for TestEncoder<'_> {
    encoder_methods! {
        emit_usize(usize);
        emit_u128(u128);
        emit_u64(u64);
        emit_u32(u32);
        emit_u16(u16);
        emit_u8(u8);
        emit_isize(isize);
        emit_i128(i128);
        emit_i64(i64);
        emit_i32(i32);
        emit_i16(i16);
    }
    fn emit_raw_bytes(&mut self, value: &[u8]) {
        self.opaque.emit_raw_bytes(value);
    }
}

impl SpanEncoder for TestEncoder<'_> {
    fn encode_span(&mut self, span: Span) {
        self.opaque.encode_span(span);
    }
    fn encode_symbol(&mut self, symbol: Symbol) {
        self.opaque.encode_symbol(symbol);
    }
    fn encode_byte_symbol(&mut self, symbol: ByteSymbol) {
        self.opaque.encode_byte_symbol(symbol);
    }
    fn encode_expn_id(&mut self, id: ExpnId) {
        self.opaque.encode_expn_id(id);
    }
    fn encode_syntax_context(&mut self, context: SyntaxContext) {
        self.opaque.encode_syntax_context(context);
    }
    fn encode_crate_num(&mut self, cnum: CrateNum) {
        self.opaque.encode_crate_num(cnum);
    }
    fn encode_def_index(&mut self, index: DefIndex) {
        self.emit_u32(index.as_u32());
    }
    fn encode_def_id(&mut self, id: DefId) {
        self.encode_crate_num(id.krate);
        self.encode_def_index(id.index);
    }
}

impl<'tcx> TyEncoder<'tcx> for TestEncoder<'tcx> {
    const CLEAR_CROSS_CRATE: bool = false;
    fn position(&self) -> usize {
        self.opaque.position()
    }
    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize> {
        &mut self.types
    }
    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<'tcx>, usize> {
        &mut self.predicates
    }
    fn trait_evidence_shorthands(&mut self) -> &mut FxHashMap<TraitEvidence<'tcx>, usize> {
        &mut self.evidence
    }
    fn encode_alloc_id(&mut self, _: &AllocId) {
        panic!("the test contains no allocations");
    }
}

struct TestDecoder<'a, 'tcx> {
    opaque: MemDecoder<'a>,
    tcx: TyCtxt<'tcx>,
    types: FxHashMap<usize, Ty<'tcx>>,
    evidence: FxHashMap<usize, TraitEvidence<'tcx>>,
    in_progress: FxHashSet<usize>,
}

impl<'a, 'tcx> TestDecoder<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, bytes: &'a [u8]) -> Self {
        Self {
            opaque: MemDecoder::new(bytes, 0).unwrap(),
            tcx,
            types: Default::default(),
            evidence: Default::default(),
            in_progress: Default::default(),
        }
    }
}

rustc_middle::implement_ty_decoder!(TestDecoder<'a, 'tcx>);

impl BlobDecoder for TestDecoder<'_, '_> {
    fn decode_symbol(&mut self) -> Symbol {
        self.opaque.decode_symbol()
    }
    fn decode_byte_symbol(&mut self) -> ByteSymbol {
        self.opaque.decode_byte_symbol()
    }
    fn decode_def_index(&mut self) -> DefIndex {
        DefIndex::from_u32(self.read_u32())
    }
}

impl SpanDecoder for TestDecoder<'_, '_> {
    fn decode_span(&mut self) -> Span {
        self.opaque.decode_span()
    }
    fn decode_expn_id(&mut self) -> ExpnId {
        self.opaque.decode_expn_id()
    }
    fn decode_syntax_context(&mut self) -> SyntaxContext {
        self.opaque.decode_syntax_context()
    }
    fn decode_crate_num(&mut self) -> CrateNum {
        self.opaque.decode_crate_num()
    }
    fn decode_def_id(&mut self) -> DefId {
        DefId { krate: self.decode_crate_num(), index: self.decode_def_index() }
    }
    fn decode_attr_id(&mut self) -> rustc_ast::AttrId {
        panic!("the test contains no attributes");
    }
}

impl<'tcx> ty::InternerDecoder for TestDecoder<'_, 'tcx> {
    type Interner = TyCtxt<'tcx>;
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> TyDecoder<'tcx> for TestDecoder<'_, 'tcx> {
    const CLEAR_CROSS_CRATE: bool = false;

    fn cached_ty_for_shorthand<F>(&mut self, shorthand: usize, f: F) -> Ty<'tcx>
    where
        F: FnOnce(&mut Self) -> Ty<'tcx>,
    {
        if let Some(&ty) = self.types.get(&shorthand) {
            return ty;
        }
        let ty = f(self);
        self.types.insert(shorthand, ty);
        ty
    }

    fn cached_trait_evidence_for_shorthand<F>(
        &mut self,
        shorthand: usize,
        f: F,
    ) -> TraitEvidence<'tcx>
    where
        F: FnOnce(&mut Self) -> TraitEvidence<'tcx>,
    {
        if let Some(&evidence) = self.evidence.get(&shorthand) {
            return evidence;
        }
        let evidence = f(self);
        self.evidence.insert(shorthand, evidence);
        evidence
    }

    fn trait_evidence_in_progress(&mut self) -> &mut FxHashSet<usize> {
        &mut self.in_progress
    }

    fn with_position<F, R>(&mut self, position: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let decoder = self.opaque.split_at(position);
        let previous = std::mem::replace(&mut self.opaque, decoder);
        let result = f(self);
        self.opaque = previous;
        result
    }

    fn decode_alloc_id(&mut self) -> AllocId {
        panic!("the test contains no allocations");
    }
}
