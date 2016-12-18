// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty::AdtKind;
use ty::layout::{Align, Size};

use rustc_data_structures::fx::{FxHashSet};

use std::cmp::{self, Ordering};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct VariantInfo {
    pub name: Option<String>,
    pub kind: SizeKind,
    pub size: u64,
    pub align: u64,
    pub fields: Vec<FieldInfo>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SizeKind { Exact, Min }

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FieldInfo {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub align: u64,
}

impl From<AdtKind> for DataTypeKind {
    fn from(kind: AdtKind) -> Self {
        match kind {
            AdtKind::Struct => DataTypeKind::Struct,
            AdtKind::Enum => DataTypeKind::Enum,
            AdtKind::Union => DataTypeKind::Union,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum DataTypeKind {
    Struct,
    Union,
    Enum,
    Closure,
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct TypeSizeInfo {
    pub kind: DataTypeKind,
    pub type_description: String,
    pub align: u64,
    pub overall_size: u64,
    pub opt_discr_size: Option<u64>,
    pub variants: Vec<VariantInfo>,
}

#[derive(PartialEq, Eq, Debug)]
pub struct CodeStats {
    type_sizes: FxHashSet<TypeSizeInfo>,
}

impl CodeStats {
    pub fn new() -> Self { CodeStats { type_sizes: FxHashSet() } }

    pub fn record_type_size<S: ToString>(&mut self,
                                         kind: DataTypeKind,
                                         type_desc: S,
                                         align: Align,
                                         overall_size: Size,
                                         opt_discr_size: Option<Size>,
                                         variants: Vec<VariantInfo>) {
        let info = TypeSizeInfo {
            kind: kind,
            type_description: type_desc.to_string(),
            align: align.abi(),
            overall_size: overall_size.bytes(),
            opt_discr_size: opt_discr_size.map(|s| s.bytes()),
            variants: variants,
        };
        self.type_sizes.insert(info);
    }

    pub fn print_type_sizes(&self) {
        let mut sorted: Vec<_> = self.type_sizes.iter().collect();

        // Primary sort: large-to-small.
        // Secondary sort: description (dictionary order)
        sorted.sort_by(|info1, info2| {
            // (reversing cmp order to get large-to-small ordering)
            match info2.overall_size.cmp(&info1.overall_size) {
                Ordering::Equal => info1.type_description.cmp(&info2.type_description),
                other => other,
            }
        });

        for info in &sorted {
            println!("print-type-size type: `{}`: {} bytes, alignment: {} bytes",
                     info.type_description, info.overall_size, info.align);
            let indent = "    ";

            let discr_size = if let Some(discr_size) = info.opt_discr_size {
                println!("print-type-size {}discriminant: {} bytes",
                         indent, discr_size);
                discr_size
            } else {
                0
            };

            // We start this at discr_size (rather than 0) because
            // things like C-enums do not have variants but we still
            // want the max_variant_size at the end of the loop below
            // to reflect the presence of the discriminant.
            let mut max_variant_size = discr_size;

            let struct_like = match info.kind {
                DataTypeKind::Struct | DataTypeKind::Closure => true,
                DataTypeKind::Enum | DataTypeKind::Union => false,
            };
            for (i, variant_info) in info.variants.iter().enumerate() {
                let VariantInfo { ref name, kind: _, align: _, size, ref fields } = *variant_info;
                let indent = if !struct_like {
                    let name = match name.as_ref() {
                        Some(name) => format!("{}", name),
                        None => format!("{}", i),
                    };
                    println!("print-type-size {}variant `{}`: {} bytes",
                             indent, name, size - discr_size);
                    "        "
                } else {
                    assert!(i < 1);
                    "    "
                };
                max_variant_size = cmp::max(max_variant_size, size);

                let mut min_offset = discr_size;

                // We want to print fields by increasing offset.
                let mut fields = fields.clone();
                fields.sort_by_key(|f| f.offset);

                for field in fields.iter() {
                    let FieldInfo { ref name, offset, size, align } = *field;

                    // Include field alignment in output only if it caused padding injection
                    if min_offset != offset {
                        let pad = offset - min_offset;
                        println!("print-type-size {}padding: {} bytes",
                                 indent, pad);
                        println!("print-type-size {}field `.{}`: {} bytes, alignment: {} bytes",
                                 indent, name, size, align);
                    } else {
                        println!("print-type-size {}field `.{}`: {} bytes",
                                 indent, name, size);
                    }

                    min_offset = offset + size;
                }
            }

            assert!(max_variant_size <= info.overall_size,
                    "max_variant_size {} !<= {} overall_size",
                    max_variant_size, info.overall_size);
            if max_variant_size < info.overall_size {
                println!("print-type-size {}end padding: {} bytes",
                         indent, info.overall_size - max_variant_size);
            }
        }
    }
}
