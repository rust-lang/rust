use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::LazyLock;

use crate::format_code;
use crate::input::InputType;
use crate::intrinsic::Intrinsic;
use crate::typekinds::BaseType;
use crate::typekinds::{ToRepr, TypeKind};

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

// Number of vectors in our buffers - the maximum tuple size, 4, plus 1 as we set the vnum
// argument to 1.
const NUM_VECS: usize = 5;
// The maximum vector length (in bits)
const VL_MAX_BITS: usize = 2048;
// The maximum vector length (in bytes)
const VL_MAX_BYTES: usize = VL_MAX_BITS / 8;
// The maximum number of elements in each vector type
const LEN_F32: usize = VL_MAX_BYTES / core::mem::size_of::<f32>();
const LEN_F64: usize = VL_MAX_BYTES / core::mem::size_of::<f64>();
const LEN_I8: usize = VL_MAX_BYTES / core::mem::size_of::<i8>();
const LEN_I16: usize = VL_MAX_BYTES / core::mem::size_of::<i16>();
const LEN_I32: usize = VL_MAX_BYTES / core::mem::size_of::<i32>();
const LEN_I64: usize = VL_MAX_BYTES / core::mem::size_of::<i64>();
const LEN_U8: usize = VL_MAX_BYTES / core::mem::size_of::<u8>();
const LEN_U16: usize = VL_MAX_BYTES / core::mem::size_of::<u16>();
const LEN_U32: usize = VL_MAX_BYTES / core::mem::size_of::<u32>();
const LEN_U64: usize = VL_MAX_BYTES / core::mem::size_of::<u64>();

/// `load_intrinsics` and `store_intrinsics` is a vector of intrinsics
/// variants, while `out_path` is a file to write to.
pub fn generate_load_store_tests(
    load_intrinsics: Vec<Intrinsic>,
    store_intrinsics: Vec<Intrinsic>,
    out_path: Option<&PathBuf>,
) -> Result<(), String> {
    let output = match out_path {
        Some(out) => {
            Box::new(File::create(out).map_err(|e| format!("couldn't create tests file: {e}"))?)
                as Box<dyn Write>
        }
        None => Box::new(std::io::stdout()) as Box<dyn Write>,
    };
    let mut used_stores = vec![false; store_intrinsics.len()];
    let tests: Vec<_> = load_intrinsics
        .iter()
        .map(|load| {
            let store_candidate = load
                .signature
                .fn_name()
                .to_string()
                .replace("svld1s", "svst1")
                .replace("svld1u", "svst1")
                .replace("svldnt1s", "svstnt1")
                .replace("svldnt1u", "svstnt1")
                .replace("svld", "svst")
                .replace("gather", "scatter");

            let store_index = store_intrinsics
                .iter()
                .position(|i| i.signature.fn_name().to_string() == store_candidate);
            if let Some(i) = store_index {
                used_stores[i] = true;
            }

            generate_single_test(
                load.clone(),
                store_index.map(|i| store_intrinsics[i].clone()),
            )
        })
        .try_collect()?;

    assert!(
        used_stores.into_iter().all(|b| b),
        "Not all store tests have been paired with a load. Consider generating specifc store-only tests"
    );

    let preamble =
        TokenStream::from_str(&PREAMBLE).map_err(|e| format!("Preamble is invalid: {e}"))?;
    // Only output manual tests for the SVE set
    let manual_tests = match &load_intrinsics[0].target_features[..] {
        [s] if s == "sve" => TokenStream::from_str(MANUAL_TESTS)
            .map_err(|e| format!("Manual tests are invalid: {e}"))?,
        _ => quote!(),
    };
    format_code(
        output,
        format!(
            "// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen-arm/spec/sve` and run the following command to re-generate
// this file:
//
// ```
// cargo run --bin=stdarch-gen-arm -- crates/stdarch-gen-arm/spec
// ```
{}",
            quote! { #preamble #(#tests)* #manual_tests }
        ),
    )
    .map_err(|e| format!("couldn't write tests: {e}"))
}

/// A test looks like this:
/// ```
///     let data = [scalable vector];
///
///     let mut storage = [0; N];
///
///     store_intrinsic([true_predicate], storage.as_mut_ptr(), data);
///     [test contents of storage]
///
///     let loaded == load_intrinsic([true_predicate], storage.as_ptr())
///     assert!(loaded == data);
/// ```
/// We intialise our data such that the value stored matches the index it's stored to.
/// By doing this we can validate scatters by checking that each value in the storage
/// array is either 0 or the same as its index.
fn generate_single_test(
    load: Intrinsic,
    store: Option<Intrinsic>,
) -> Result<proc_macro2::TokenStream, String> {
    let chars = LdIntrCharacteristics::new(&load)?;
    let fn_name = load.signature.fn_name().to_string();

    #[allow(clippy::collapsible_if)]
    if let Some(ty) = &chars.gather_bases_type {
        if ty.base_type().unwrap().get_size() == Ok(32)
            && chars.gather_index_type.is_none()
            && chars.gather_offset_type.is_none()
        {
            // We lack a way to ensure data is in the bottom 32 bits of the address space
            println!("Skipping test for {fn_name}");
            return Ok(quote!());
        }
    }

    if fn_name.starts_with("svldff1") && fn_name.contains("gather") {
        // TODO: We can remove this check when first-faulting gathers are fixed in CI's QEMU
        // https://gitlab.com/qemu-project/qemu/-/issues/1612
        println!("Skipping test for {fn_name}");
        return Ok(quote!());
    }

    let fn_ident = format_ident!("{fn_name}");
    let test_name = format_ident!(
        "test_{fn_name}{}",
        if let Some(ref store) = store {
            format!("_with_{}", store.signature.fn_name())
        } else {
            String::new()
        }
    );

    let load_type = &chars.load_type;
    let acle_type = load_type.acle_notation_repr();

    // If there's no return type, fallback to the load type for things that depend on it
    let ret_type = &load
        .signature
        .return_type
        .as_ref()
        .and_then(TypeKind::base_type)
        .unwrap_or(load_type);

    let pred_fn = format_ident!("svptrue_b{}", load_type.size());

    let load_type_caps = load_type.rust_repr().to_uppercase();
    let data_array = format_ident!("{load_type_caps}_DATA");

    let size_fn = format_ident!("svcnt{}", ret_type.size_literal());

    let rust_ret_type = ret_type.rust_repr();
    let assert_fn = format_ident!("assert_vector_matches_{rust_ret_type}");

    // Use vnum=1, so adjust all values by one vector length
    let (length_call, vnum_arg) = if chars.vnum {
        if chars.is_prf {
            (quote!(), quote!(, 1))
        } else {
            (quote!(let len = #size_fn() as usize;), quote!(, 1))
        }
    } else {
        (quote!(), quote!())
    };

    let (bases_load, bases_arg) = if let Some(ty) = &chars.gather_bases_type {
        // Bases is a vector of (sometimes 32-bit) pointers
        // When we combine bases with an offset/index argument, we load from the data arrays
        // starting at 1
        let base_ty = ty.base_type().unwrap();
        let rust_type = format_ident!("{}", base_ty.rust_repr());
        let index_fn = format_ident!("svindex_{}", base_ty.acle_notation_repr());
        let size_in_bytes = chars.load_type.get_size().unwrap() / 8;

        if base_ty.get_size().unwrap() == 32 {
            // Treat bases as a vector of offsets here - we don't test this without an offset or
            // index argument
            (
                Some(quote!(
                    let bases = #index_fn(0, #size_in_bytes.try_into().unwrap());
                )),
                quote!(, bases),
            )
        } else {
            // Treat bases as a vector of pointers
            let base_fn = format_ident!("svdup_n_{}", base_ty.acle_notation_repr());
            let data_array = if store.is_some() {
                format_ident!("storage")
            } else {
                format_ident!("{}_DATA", chars.load_type.rust_repr().to_uppercase())
            };

            let add_fn = format_ident!("svadd_{}_x", base_ty.acle_notation_repr());
            (
                Some(quote! {
                    let bases = #base_fn(#data_array.as_ptr() as #rust_type);
                    let offsets = #index_fn(0, #size_in_bytes.try_into().unwrap());
                    let bases = #add_fn(#pred_fn(), bases, offsets);
                }),
                quote!(, bases),
            )
        }
    } else {
        (None, quote!())
    };

    let index_arg = if let Some(ty) = &chars.gather_index_type {
        let rust_type = format_ident!("{}", ty.rust_repr());
        if chars
            .gather_bases_type
            .as_ref()
            .and_then(TypeKind::base_type)
            .map_or(Err(String::new()), BaseType::get_size)
            .unwrap()
            == 32
        {
            // Let index be the base of the data array
            let data_array = if store.is_some() {
                format_ident!("storage")
            } else {
                format_ident!("{}_DATA", chars.load_type.rust_repr().to_uppercase())
            };
            let size_in_bytes = chars.load_type.get_size().unwrap() / 8;
            quote!(, #data_array.as_ptr() as #rust_type / (#size_in_bytes as #rust_type) + 1)
        } else {
            quote!(, 1.try_into().unwrap())
        }
    } else {
        quote!()
    };

    let offset_arg = if let Some(ty) = &chars.gather_offset_type {
        let size_in_bytes = chars.load_type.get_size().unwrap() / 8;
        if chars
            .gather_bases_type
            .as_ref()
            .and_then(TypeKind::base_type)
            .map_or(Err(String::new()), BaseType::get_size)
            .unwrap()
            == 32
        {
            // Let offset be the base of the data array
            let rust_type = format_ident!("{}", ty.rust_repr());
            let data_array = if store.is_some() {
                format_ident!("storage")
            } else {
                format_ident!("{}_DATA", chars.load_type.rust_repr().to_uppercase())
            };
            quote!(, #data_array.as_ptr() as #rust_type + #size_in_bytes as #rust_type)
        } else {
            quote!(, #size_in_bytes.try_into().unwrap())
        }
    } else {
        quote!()
    };

    let (offsets_load, offsets_arg) = if let Some(ty) = &chars.gather_offsets_type {
        // Offsets is a scalable vector of per-element offsets in bytes. We re-use the contiguous
        // data for this, then multiply to get indices
        let offsets_fn = format_ident!("svindex_{}", ty.base_type().unwrap().acle_notation_repr());
        let size_in_bytes = chars.load_type.get_size().unwrap() / 8;
        (
            Some(quote! {
                let offsets = #offsets_fn(0, #size_in_bytes.try_into().unwrap());
            }),
            quote!(, offsets),
        )
    } else {
        (None, quote!())
    };

    let (indices_load, indices_arg) = if let Some(ty) = &chars.gather_indices_type {
        // There's no need to multiply indices by the load type width
        let base_ty = ty.base_type().unwrap();
        let indices_fn = format_ident!("svindex_{}", base_ty.acle_notation_repr());
        (
            Some(quote! {
                let indices = #indices_fn(0, 1);
            }),
            quote! {, indices},
        )
    } else {
        (None, quote!())
    };

    let ptr = if chars.gather_bases_type.is_some() {
        quote!()
    } else if chars.is_prf {
        quote!(, I64_DATA.as_ptr())
    } else {
        quote!(, #data_array.as_ptr())
    };

    let tuple_len = &chars.tuple_len;
    let expecteds = if chars.is_prf {
        // No return value for prefetches
        vec![]
    } else {
        (0..*tuple_len)
            .map(|i| get_expected_range(i, &chars))
            .collect()
    };
    let asserts: Vec<_> =
        if *tuple_len > 1 {
            let svget = format_ident!("svget{tuple_len}_{acle_type}");
            expecteds.iter().enumerate().map(|(i, expected)| {
            quote! (#assert_fn(#svget::<{ #i as i32 }>(loaded), #expected);)
        }).collect()
        } else {
            expecteds
                .iter()
                .map(|expected| quote! (#assert_fn(loaded, #expected);))
                .collect()
        };

    let function = if chars.is_prf {
        if fn_name.contains("gather") && fn_name.contains("base") && !fn_name.starts_with("svprf_")
        {
            // svprf(b|h|w|d)_gather base intrinsics do not have a generic type parameter
            quote!(#fn_ident::<{ svprfop::SV_PLDL1KEEP }>)
        } else {
            quote!(#fn_ident::<{ svprfop::SV_PLDL1KEEP }, i64>)
        }
    } else {
        quote!(#fn_ident)
    };

    let octaword_guard = if chars.replicate_width == Some(256) {
        let msg = format!("Skipping {test_name} due to SVE vector length");
        quote! {
            if svcntb() < 32 {
                println!(#msg);
                return;
            }
        }
    } else {
        quote!()
    };

    let feats = load.target_features.join(",");

    if let Some(store) = store {
        let data_init = if *tuple_len == 1 {
            quote!(#(#expecteds)*)
        } else {
            let create = format_ident!("svcreate{tuple_len}_{acle_type}");
            quote!(#create(#(#expecteds),*))
        };
        let input = store.input.types.first().unwrap().get(0).unwrap();
        let store_type = input
            .get(store.test.get_typeset_index().unwrap())
            .and_then(InputType::typekind)
            .and_then(TypeKind::base_type)
            .unwrap();

        let store_type = format_ident!("{}", store_type.rust_repr());
        let storage_len = NUM_VECS * VL_MAX_BITS / chars.load_type.get_size()? as usize;
        let store_fn = format_ident!("{}", store.signature.fn_name().to_string());
        let load_type = format_ident!("{}", chars.load_type.rust_repr());
        let (store_ptr, store_mut_ptr) = if chars.gather_bases_type.is_none() {
            (
                quote!(, storage.as_ptr() as *const #load_type),
                quote!(, storage.as_mut_ptr()),
            )
        } else {
            (quote!(), quote!())
        };
        let args = quote!(#pred_fn() #store_ptr #vnum_arg #bases_arg #offset_arg #index_arg #offsets_arg #indices_arg);
        let call = if chars.uses_ffr {
            // Doing a normal load first maximises the number of elements our ff/nf test loads
            let non_ffr_fn_name = format_ident!(
                "{}",
                fn_name
                    .replace("svldff1", "svld1")
                    .replace("svldnf1", "svld1")
            );
            quote! {
                svsetffr();
                let _ = #non_ffr_fn_name(#args);
                let loaded = #function(#args);
            }
        } else {
            // Note that the FFR must be set for all tests as the assert functions mask against it
            quote! {
                svsetffr();
                let loaded = #function(#args);
            }
        };

        Ok(quote! {
            #[simd_test(enable = #feats)]
            unsafe fn #test_name() {
                #octaword_guard
                #length_call
                let mut storage = [0 as #store_type; #storage_len];
                let data = #data_init;
                #bases_load
                #offsets_load
                #indices_load

                #store_fn(#pred_fn() #store_mut_ptr #vnum_arg #bases_arg #offset_arg #index_arg #offsets_arg #indices_arg, data);
                for (i, &val) in storage.iter().enumerate() {
                    assert!(val == 0 as #store_type || val == i as #store_type);
                }

                #call
                #(#asserts)*

            }
        })
    } else {
        let args = quote!(#pred_fn() #ptr #vnum_arg #bases_arg #offset_arg #index_arg #offsets_arg #indices_arg);
        let call = if chars.uses_ffr {
            // Doing a normal load first maximises the number of elements our ff/nf test loads
            let non_ffr_fn_name = format_ident!(
                "{}",
                fn_name
                    .replace("svldff1", "svld1")
                    .replace("svldnf1", "svld1")
            );
            quote! {
                svsetffr();
                let _ = #non_ffr_fn_name(#args);
                let loaded = #function(#args);
            }
        } else {
            // Note that the FFR must be set for all tests as the assert functions mask against it
            quote! {
                svsetffr();
                let loaded = #function(#args);
            }
        };
        Ok(quote! {
            #[simd_test(enable = #feats)]
            unsafe fn #test_name() {
                #octaword_guard
                #bases_load
                #offsets_load
                #indices_load
                #call
                #length_call

                #(#asserts)*
            }
        })
    }
}

/// Assumes chars.ret_type is not None
fn get_expected_range(tuple_idx: usize, chars: &LdIntrCharacteristics) -> proc_macro2::TokenStream {
    // vnum=1
    let vnum_adjust = if chars.vnum { quote!(len+) } else { quote!() };

    let bases_adjust =
        (chars.gather_index_type.is_some() || chars.gather_offset_type.is_some()) as usize;

    let tuple_len = chars.tuple_len;
    let size = chars
        .ret_type
        .as_ref()
        .and_then(TypeKind::base_type)
        .unwrap_or(&chars.load_type)
        .get_size()
        .unwrap() as usize;

    if chars.replicate_width == Some(128) {
        // svld1rq
        let ty_rust = format_ident!(
            "{}",
            chars
                .ret_type
                .as_ref()
                .unwrap()
                .base_type()
                .unwrap()
                .rust_repr()
        );
        let args: Vec<_> = (0..(128 / size)).map(|i| quote!(#i as #ty_rust)).collect();
        let dup = format_ident!(
            "svdupq_n_{}",
            chars.ret_type.as_ref().unwrap().acle_notation_repr()
        );
        quote!(#dup(#(#args,)*))
    } else if chars.replicate_width == Some(256) {
        // svld1ro - we use two interleaved svdups to create a repeating 256-bit pattern
        let ty_rust = format_ident!(
            "{}",
            chars
                .ret_type
                .as_ref()
                .unwrap()
                .base_type()
                .unwrap()
                .rust_repr()
        );
        let ret_acle = chars.ret_type.as_ref().unwrap().acle_notation_repr();
        let args: Vec<_> = (0..(128 / size)).map(|i| quote!(#i as #ty_rust)).collect();
        let args2: Vec<_> = ((128 / size)..(256 / size))
            .map(|i| quote!(#i as #ty_rust))
            .collect();
        let dup = format_ident!("svdupq_n_{ret_acle}");
        let interleave = format_ident!("svtrn1q_{ret_acle}");
        quote!(#interleave(#dup(#(#args,)*), #dup(#(#args2,)*)))
    } else {
        let start = bases_adjust + tuple_idx;
        if chars
            .ret_type
            .as_ref()
            .unwrap()
            .base_type()
            .unwrap()
            .is_float()
        {
            // Use svcvt to create a linear sequence of floats
            let cvt_fn = format_ident!("svcvt_f{size}_s{size}_x");
            let pred_fn = format_ident!("svptrue_b{size}");
            let svindex_fn = format_ident!("svindex_s{size}");
            quote! { #cvt_fn(#pred_fn(), #svindex_fn((#vnum_adjust #start).try_into().unwrap(), #tuple_len.try_into().unwrap()))}
        } else {
            let ret_acle = chars.ret_type.as_ref().unwrap().acle_notation_repr();
            let svindex = format_ident!("svindex_{ret_acle}");
            quote!(#svindex((#vnum_adjust #start).try_into().unwrap(), #tuple_len.try_into().unwrap()))
        }
    }
}

struct LdIntrCharacteristics {
    // The data type to load from (not necessarily the data type returned)
    load_type: BaseType,
    // The data type to return (None for unit)
    ret_type: Option<TypeKind>,
    // The size of tuple to load/store
    tuple_len: usize,
    // Whether a vnum argument is present
    vnum: bool,
    // Is the intrinsic first/non-faulting?
    uses_ffr: bool,
    // Is it a prefetch?
    is_prf: bool,
    // The size of data loaded with svld1ro/q intrinsics
    replicate_width: Option<usize>,
    // Scalable vector of pointers to load from
    gather_bases_type: Option<TypeKind>,
    // Scalar offset, paired with bases
    gather_offset_type: Option<TypeKind>,
    // Scalar index, paired with bases
    gather_index_type: Option<TypeKind>,
    // Scalable vector of offsets
    gather_offsets_type: Option<TypeKind>,
    // Scalable vector of indices
    gather_indices_type: Option<TypeKind>,
}

impl LdIntrCharacteristics {
    fn new(intr: &Intrinsic) -> Result<LdIntrCharacteristics, String> {
        let input = intr.input.types.first().unwrap().get(0).unwrap();
        let load_type = input
            .get(intr.test.get_typeset_index().unwrap())
            .and_then(InputType::typekind)
            .and_then(TypeKind::base_type)
            .unwrap();

        let ret_type = intr.signature.return_type.clone();

        let name = intr.signature.fn_name().to_string();
        let tuple_len = name
            .chars()
            .find(|c| c.is_numeric())
            .and_then(|c| c.to_digit(10))
            .unwrap_or(1) as usize;

        let uses_ffr = name.starts_with("svldff") || name.starts_with("svldnf");

        let is_prf = name.starts_with("svprf");

        let replicate_width = if name.starts_with("svld1ro") {
            Some(256)
        } else if name.starts_with("svld1rq") {
            Some(128)
        } else {
            None
        };

        let get_ty_of_arg = |name: &str| {
            intr.signature
                .arguments
                .iter()
                .find(|a| a.name.to_string() == name)
                .map(|a| a.kind.clone())
        };

        let gather_bases_type = get_ty_of_arg("bases");
        let gather_offset_type = get_ty_of_arg("offset");
        let gather_index_type = get_ty_of_arg("index");
        let gather_offsets_type = get_ty_of_arg("offsets");
        let gather_indices_type = get_ty_of_arg("indices");

        Ok(LdIntrCharacteristics {
            load_type: *load_type,
            ret_type,
            tuple_len,
            vnum: name.contains("vnum"),
            uses_ffr,
            is_prf,
            replicate_width,
            gather_bases_type,
            gather_offset_type,
            gather_index_type,
            gather_offsets_type,
            gather_indices_type,
        })
    }
}

static PREAMBLE: LazyLock<String> = LazyLock::new(|| {
    format!(
        r#"#![allow(unused)]

use super::*;
use std::boxed::Box;
use std::convert::{{TryFrom, TryInto}};
use std::sync::LazyLock;
use std::vec::Vec;
use stdarch_test::simd_test;

static F32_DATA: LazyLock<[f32; {LEN_F32} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_F32} * {NUM_VECS})
        .map(|i| i as f32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("f32 data incorrectly initialised")
}});
static F64_DATA: LazyLock<[f64; {LEN_F64} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_F64} * {NUM_VECS})
        .map(|i| i as f64)
        .collect::<Vec<_>>()
        .try_into()
        .expect("f64 data incorrectly initialised")
}});
static I8_DATA: LazyLock<[i8; {LEN_I8} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_I8} * {NUM_VECS})
        .map(|i| ((i + 128) % 256 - 128) as i8)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i8 data incorrectly initialised")
}});
static I16_DATA: LazyLock<[i16; {LEN_I16} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_I16} * {NUM_VECS})
        .map(|i| i as i16)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i16 data incorrectly initialised")
}});
static I32_DATA: LazyLock<[i32; {LEN_I32} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_I32} * {NUM_VECS})
        .map(|i| i as i32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i32 data incorrectly initialised")
}});
static I64_DATA: LazyLock<[i64; {LEN_I64} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_I64} * {NUM_VECS})
        .map(|i| i as i64)
        .collect::<Vec<_>>()
        .try_into()
        .expect("i64 data incorrectly initialised")
}});
static U8_DATA: LazyLock<[u8; {LEN_U8} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_U8} * {NUM_VECS})
        .map(|i| i as u8)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u8 data incorrectly initialised")
}});
static U16_DATA: LazyLock<[u16; {LEN_U16} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_U16} * {NUM_VECS})
        .map(|i| i as u16)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u16 data incorrectly initialised")
}});
static U32_DATA: LazyLock<[u32; {LEN_U32} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_U32} * {NUM_VECS})
        .map(|i| i as u32)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u32 data incorrectly initialised")
}});
static U64_DATA: LazyLock<[u64; {LEN_U64} * {NUM_VECS}]> = LazyLock::new(|| {{
    (0..{LEN_U64} * {NUM_VECS})
        .map(|i| i as u64)
        .collect::<Vec<_>>()
        .try_into()
        .expect("u64 data incorrectly initialised")
}});

#[target_feature(enable = "sve")]
fn assert_vector_matches_f32(vector: svfloat32_t, expected: svfloat32_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b32(), defined));
    let cmp = svcmpne_f32(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_f64(vector: svfloat64_t, expected: svfloat64_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b64(), defined));
    let cmp = svcmpne_f64(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_i8(vector: svint8_t, expected: svint8_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b8(), defined));
    let cmp = svcmpne_s8(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_i16(vector: svint16_t, expected: svint16_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b16(), defined));
    let cmp = svcmpne_s16(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_i32(vector: svint32_t, expected: svint32_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b32(), defined));
    let cmp = svcmpne_s32(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_i64(vector: svint64_t, expected: svint64_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b64(), defined));
    let cmp = svcmpne_s64(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_u8(vector: svuint8_t, expected: svuint8_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b8(), defined));
    let cmp = svcmpne_u8(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_u16(vector: svuint16_t, expected: svuint16_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b16(), defined));
    let cmp = svcmpne_u16(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_u32(vector: svuint32_t, expected: svuint32_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b32(), defined));
    let cmp = svcmpne_u32(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}

#[target_feature(enable = "sve")]
fn assert_vector_matches_u64(vector: svuint64_t, expected: svuint64_t) {{
    let defined = svrdffr();
    assert!(svptest_first(svptrue_b64(), defined));
    let cmp = svcmpne_u64(defined, vector, expected);
    assert!(!svptest_any(defined, cmp))
}}
"#
    )
});

const MANUAL_TESTS: &str = "#[simd_test(enable = \"sve\")]
unsafe fn test_ffr() {
    svsetffr();
    let ffr = svrdffr();
    assert_vector_matches_u8(svdup_n_u8_z(ffr, 1), svindex_u8(1, 0));
    let pred = svdupq_n_b8(true, false, true, false, true, false, true, false,
                           true, false, true, false, true, false, true, false);
    svwrffr(pred);
    let ffr = svrdffr_z(svptrue_b8());
    assert_vector_matches_u8(svdup_n_u8_z(ffr, 1), svdup_n_u8_z(pred, 1));
}
";
