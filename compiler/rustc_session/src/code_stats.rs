use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lock;
use rustc_span::def_id::DefId;
use rustc_span::Symbol;
use rustc_target::abi::{Align, Size};
use std::cmp;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VariantInfo {
    pub name: Option<Symbol>,
    pub kind: SizeKind,
    pub size: u64,
    pub align: u64,
    pub fields: Vec<FieldInfo>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SizeKind {
    Exact,
    Min,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FieldKind {
    AdtField,
    Upvar,
    GeneratorLocal,
}

impl std::fmt::Display for FieldKind {
    fn fmt(&self, w: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldKind::AdtField => write!(w, "field"),
            FieldKind::Upvar => write!(w, "upvar"),
            FieldKind::GeneratorLocal => write!(w, "local"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FieldInfo {
    pub kind: FieldKind,
    pub name: Symbol,
    pub offset: u64,
    pub size: u64,
    pub align: u64,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum DataTypeKind {
    Struct,
    Union,
    Enum,
    Closure,
    Generator,
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct TypeSizeInfo {
    pub kind: DataTypeKind,
    pub type_description: String,
    pub align: u64,
    pub overall_size: u64,
    pub packed: bool,
    pub opt_discr_size: Option<u64>,
    pub variants: Vec<VariantInfo>,
}

pub struct VTableSizeInfo {
    pub trait_name: String,

    /// Number of entries in a vtable with the current algorithm
    /// (i.e. with upcasting).
    pub entries: usize,

    /// Number of entries in a vtable, as-if we did not have trait upcasting.
    pub entries_ignoring_upcasting: usize,

    /// Number of entries in a vtable needed solely for upcasting
    /// (i.e. `entries - entries_ignoring_upcasting`).
    pub entries_for_upcasting: usize,

    /// Cost of having upcasting in % relative to the number of entries without
    /// upcasting (i.e. `entries_for_upcasting / entries_ignoring_upcasting * 100%`).
    pub upcasting_cost_percent: f64,
}

#[derive(Default)]
pub struct CodeStats {
    type_sizes: Lock<FxHashSet<TypeSizeInfo>>,
    vtable_sizes: Lock<FxHashMap<DefId, VTableSizeInfo>>,
}

impl CodeStats {
    pub fn record_type_size<S: ToString>(
        &self,
        kind: DataTypeKind,
        type_desc: S,
        align: Align,
        overall_size: Size,
        packed: bool,
        opt_discr_size: Option<Size>,
        mut variants: Vec<VariantInfo>,
    ) {
        // Sort variants so the largest ones are shown first. A stable sort is
        // used here so that source code order is preserved for all variants
        // that have the same size.
        // Except for Generators, whose variants are already sorted according to
        // their yield points in `variant_info_for_generator`.
        if kind != DataTypeKind::Generator {
            variants.sort_by_key(|info| cmp::Reverse(info.size));
        }
        let info = TypeSizeInfo {
            kind,
            type_description: type_desc.to_string(),
            align: align.bytes(),
            overall_size: overall_size.bytes(),
            packed,
            opt_discr_size: opt_discr_size.map(|s| s.bytes()),
            variants,
        };
        self.type_sizes.borrow_mut().insert(info);
    }

    pub fn record_vtable_size(&self, trait_did: DefId, trait_name: &str, info: VTableSizeInfo) {
        let prev = self.vtable_sizes.lock().insert(trait_did, info);
        assert!(
            prev.is_none(),
            "size of vtable for `{trait_name}` ({trait_did:?}) is already recorded"
        );
    }

    pub fn print_type_sizes(&self) {
        let type_sizes = self.type_sizes.borrow();
        let mut sorted: Vec<_> = type_sizes.iter().collect();

        // Primary sort: large-to-small.
        // Secondary sort: description (dictionary order)
        sorted.sort_by_key(|info| (cmp::Reverse(info.overall_size), &info.type_description));

        for info in sorted {
            let TypeSizeInfo { type_description, overall_size, align, kind, variants, .. } = info;
            println!(
                "print-type-size type: `{type_description}`: {overall_size} bytes, alignment: {align} bytes"
            );
            let indent = "    ";

            let discr_size = if let Some(discr_size) = info.opt_discr_size {
                println!("print-type-size {indent}discriminant: {discr_size} bytes");
                discr_size
            } else {
                0
            };

            // We start this at discr_size (rather than 0) because
            // things like C-enums do not have variants but we still
            // want the max_variant_size at the end of the loop below
            // to reflect the presence of the discriminant.
            let mut max_variant_size = discr_size;

            let struct_like = match kind {
                DataTypeKind::Struct | DataTypeKind::Closure => true,
                DataTypeKind::Enum | DataTypeKind::Union | DataTypeKind::Generator => false,
            };
            for (i, variant_info) in variants.into_iter().enumerate() {
                let VariantInfo { ref name, kind: _, align: _, size, ref fields } = *variant_info;
                let indent = if !struct_like {
                    let name = match name.as_ref() {
                        Some(name) => name.to_string(),
                        None => i.to_string(),
                    };
                    println!(
                        "print-type-size {indent}variant `{name}`: {diff} bytes",
                        diff = size - discr_size
                    );
                    "        "
                } else {
                    assert!(i < 1);
                    "    "
                };
                max_variant_size = cmp::max(max_variant_size, size);

                let mut min_offset = discr_size;

                // We want to print fields by increasing offset. We also want
                // zero-sized fields before non-zero-sized fields, otherwise
                // the loop below goes wrong; hence the `f.size` in the sort
                // key.
                let mut fields = fields.clone();
                fields.sort_by_key(|f| (f.offset, f.size));

                for field in fields {
                    let FieldInfo { kind, ref name, offset, size, align } = field;

                    if offset > min_offset {
                        let pad = offset - min_offset;
                        println!("print-type-size {indent}padding: {pad} bytes");
                    }

                    if offset < min_offset {
                        // If this happens it's probably a union.
                        println!(
                            "print-type-size {indent}{kind} `.{name}`: {size} bytes, \
                                  offset: {offset} bytes, \
                                  alignment: {align} bytes"
                        );
                    } else if info.packed || offset == min_offset {
                        println!("print-type-size {indent}{kind} `.{name}`: {size} bytes");
                    } else {
                        // Include field alignment in output only if it caused padding injection
                        println!(
                            "print-type-size {indent}{kind} `.{name}`: {size} bytes, \
                                  alignment: {align} bytes"
                        );
                    }

                    min_offset = offset + size;
                }
            }

            match overall_size.checked_sub(max_variant_size) {
                None => panic!("max_variant_size {max_variant_size} > {overall_size} overall_size"),
                Some(diff @ 1..) => println!("print-type-size {indent}end padding: {diff} bytes"),
                Some(0) => {}
            }
        }
    }

    pub fn print_vtable_sizes(&self, crate_name: &str) {
        let mut infos =
            std::mem::take(&mut *self.vtable_sizes.lock()).into_values().collect::<Vec<_>>();

        // Primary sort: cost % in reverse order (from largest to smallest)
        // Secondary sort: trait_name
        infos.sort_by(|a, b| {
            a.upcasting_cost_percent
                .total_cmp(&b.upcasting_cost_percent)
                .reverse()
                .then_with(|| a.trait_name.cmp(&b.trait_name))
        });

        for VTableSizeInfo {
            trait_name,
            entries,
            entries_ignoring_upcasting,
            entries_for_upcasting,
            upcasting_cost_percent,
        } in infos
        {
            println!(
                r#"print-vtable-sizes {{ "crate_name": "{crate_name}", "trait_name": "{trait_name}", "entries": "{entries}", "entries_ignoring_upcasting": "{entries_ignoring_upcasting}", "entries_for_upcasting": "{entries_for_upcasting}", "upcasting_cost_percent": "{upcasting_cost_percent}" }}"#
            );
        }
    }
}
