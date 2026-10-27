//! A macro to implement `HirDatabase` for any (`Sized`) type implementing `SourceDatabase`.

#[macro_export]
macro_rules! impl_hir_database {
    ($ty:ty) => {
        const _: () = {
            use $crate::{
                __private::{hir_def::ModuleId, salsa},
                db::HirDatabase,
                display::{DisplayTarget, HirDisplay},
                next_solver::Ty,
            };

            #[salsa::db]
            impl $crate::db::HirDatabase for $ty {
                fn as_dyn(&self) -> &dyn HirDatabase {
                    self
                }

                fn type_name<'db>(&'db self, ty: Ty<'db>, module: ModuleId) -> String {
                    match ty.display_source_code(self, module, true) {
                        Ok(ty_name) => ty_name,
                        // Fallback to human readable display in case of `Err`. Ideally we want to use `display_source_code` to
                        // render full paths.
                        Err(_) => {
                            let krate = module.krate(self);
                            ty.display(self, DisplayTarget::from_crate(self, krate)).to_string()
                        }
                    }
                }
            }
        };
    };
}
