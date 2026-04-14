use crate::ast;
use crate::ir;
use indexmap::IndexMap;
use std::collections::HashMap;

pub struct Resolver {
    kinds: IndexMap<String, ir::ResolvedKind>,
    // Mapping from dotted name (as string) to canonical name
    name_map: HashMap<String, String>,
}

impl Resolver {
    pub fn new() -> Self {
        let mut kinds = IndexMap::new();
        let mut name_map = HashMap::new();

        // Register built-ins
        let builtins = [
            ("bool", ir::BuiltinKind::Bool),
            ("u8", ir::BuiltinKind::U8),
            ("u16", ir::BuiltinKind::U16),
            ("u32", ir::BuiltinKind::U32),
            ("u64", ir::BuiltinKind::U64),
            ("i8", ir::BuiltinKind::I8),
            ("i16", ir::BuiltinKind::I16),
            ("i32", ir::BuiltinKind::I32),
            ("i64", ir::BuiltinKind::I64),
            ("string", ir::BuiltinKind::String),
            ("bytes", ir::BuiltinKind::Bytes),
            ("option", ir::BuiltinKind::Option),
            ("result", ir::BuiltinKind::Result),
            ("list", ir::BuiltinKind::List),
            ("ref", ir::BuiltinKind::Ref),
        ];

        for (name, builtin) in builtins {
            let canonical = name.to_string();
            kinds.insert(
                canonical.clone(),
                ir::ResolvedKind {
                    canonical_name: canonical.clone(),
                    doc: None,
                    kind_id: [0; 16], // Builtins don't need a KindId or have a fixed one?
                    shape: ir::KindShape::Builtin(builtin),
                },
            );
            name_map.insert(canonical.clone(), canonical);
        }

        Self { kinds, name_map }
    }

    pub fn resolve_files(&mut self, files: &[ast::File]) -> Result<ir::Schema, String> {
        // Pass 1: Register all types
        for file in files {
            for decl in &file.declarations {
                let canonical = decl.name.join(".");
                if self.kinds.contains_key(&canonical) {
                    // Allow opaque declarations for already defined kinds (e.g. builtins in prelude)
                    if decl.body.is_none() {
                        continue;
                    }
                    return Err(format!("Duplicate kind definition: {} at {:?}", canonical, decl.span));
                }
                // Placeholder for now
                self.kinds.insert(
                    canonical.clone(),
                    ir::ResolvedKind {
                        canonical_name: canonical.clone(),
                        doc: decl.doc.clone(),
                        kind_id: [0; 16],
                        shape: ir::KindShape::Builtin(ir::BuiltinKind::Bool), // Dummy
                    },
                );
                self.name_map.insert(canonical.clone(), canonical);
            }
        }

        // Pass 2: Resolve bodies and KindIds
        for file in files {
            for decl in &file.declarations {
                let canonical = decl.name.join(".");
                if let Some(body) = &decl.body {
                    let shape = match body {
                        ast::KindBody::Struct(fields) => {
                            let mut resolved_fields = Vec::new();
                            for f in fields {
                                resolved_fields.push(ir::ResolvedField {
                                    name: f.name.clone(),
                                    doc: f.doc.clone(),
                                    ty: self.resolve_type(&f.ty)?,
                                });
                            }
                            ir::KindShape::Struct(resolved_fields)
                        }
                        ast::KindBody::Enum(variants) => {
                            let mut resolved_variants = Vec::new();
                            for v in variants {
                                let payload = match &v.payload {
                                    ast::VariantPayload::Unit => ir::ResolvedVariantPayload::Unit,
                                    ast::VariantPayload::Tuple(ty) => {
                                        ir::ResolvedVariantPayload::Tuple(self.resolve_type(ty)?)
                                    }
                                    ast::VariantPayload::Struct(fields) => {
                                        let mut resolved_fields = Vec::new();
                                        for f in fields {
                                            resolved_fields.push(ir::ResolvedField {
                                                name: f.name.clone(),
                                                doc: f.doc.clone(),
                                                ty: self.resolve_type(&f.ty)?,
                                            });
                                        }
                                        ir::ResolvedVariantPayload::Struct(resolved_fields)
                                    }
                                };
                                resolved_variants.push(ir::ResolvedVariant {
                                    name: v.name.clone(),
                                    doc: v.doc.clone(),
                                    payload,
                                });
                            }
                            ir::KindShape::Enum(resolved_variants)
                        }
                        ast::KindBody::Alias(ty) => {
                            ir::KindShape::Alias(self.resolve_type(ty)?)
                        }
                    };

                    let kind_id = self.compute_kind_id_with_shape(&canonical, &shape);
                    let kind = self.kinds.get_mut(&canonical).unwrap();
                    kind.shape = shape;
                    kind.kind_id = kind_id;
                }
            }
        }

        Ok(ir::Schema {
            version: files.get(0).and_then(|f| f.version).unwrap_or(0),
            kinds: self.kinds.clone(),
        })
    }

    fn resolve_type(&self, ty: &ast::TypeExpr) -> Result<ir::ResolvedType, String> {
        let name = ty.name.join(".");
        if !self.kinds.contains_key(&name) {
            return Err(format!("Undefined type: {} at {:?}", name, ty.span));
        }

        let mut args = Vec::new();
        for arg in &ty.args {
            args.push(self.resolve_type(arg)?);
        }

        Ok(ir::ResolvedType {
            kind_ref: name,
            args,
        })
    }

    fn compute_kind_id_with_shape(&self, canonical_name: &str, shape: &ir::KindShape) -> [u8; 16] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ThingOS-KindId-v1");
        hasher.update(canonical_name.as_bytes());

        // Mix in structural info
        match shape {
            ir::KindShape::Builtin(b) => {
                hasher.update(&[0]);
                hasher.update(&[*b as u8]);
            }
            ir::KindShape::Struct(fields) => {
                hasher.update(&[1]);
                for f in fields {
                    hasher.update(f.name.as_bytes());
                    self.hash_type(&mut hasher, &f.ty);
                }
            }
            ir::KindShape::Enum(variants) => {
                hasher.update(&[2]);
                for v in variants {
                    hasher.update(v.name.as_bytes());
                    match &v.payload {
                        ir::ResolvedVariantPayload::Unit => { hasher.update(&[0]); },
                        ir::ResolvedVariantPayload::Tuple(ty) => {
                            hasher.update(&[1]);
                            self.hash_type(&mut hasher, ty);
                        }
                        ir::ResolvedVariantPayload::Struct(fields) => {
                            hasher.update(&[2]);
                            for f in fields {
                                hasher.update(f.name.as_bytes());
                                self.hash_type(&mut hasher, &f.ty);
                            }
                        }
                    }
                }
            }
            ir::KindShape::Alias(ty) => {
                hasher.update(&[3]);
                self.hash_type(&mut hasher, ty);
            }
        }

        let hash = hasher.finalize();
        let mut out = [0u8; 16];
        out.copy_from_slice(&hash.as_bytes()[0..16]);
        out
    }

    fn compute_kind_id(&self, kind: &ir::ResolvedKind) -> [u8; 16] {
        self.compute_kind_id_with_shape(&kind.canonical_name, &kind.shape)
    }

    fn hash_type(&self, hasher: &mut blake3::Hasher, ty: &ir::ResolvedType) {
        hasher.update(ty.kind_ref.as_bytes());
        for arg in &ty.args {
            self.hash_type(hasher, arg);
        }
    }
}
