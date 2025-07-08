//! See [`FamousDefs`].

use base_db::{CrateOrigin, LangCrateOrigin};
use hir::{Crate, Enum, Function, Macro, Module, ScopeDef, Semantics, Trait};

use crate::RootDatabase;

/// Helps with finding well-know things inside the standard library. This is
/// somewhat similar to the known paths infra inside hir, but it different; We
/// want to make sure that IDE specific paths don't become interesting inside
/// the compiler itself as well.
///
/// Note that, by default, rust-analyzer tests **do not** include core or std
/// libraries. If you are writing tests for functionality using [`FamousDefs`],
/// you'd want to include minicore (see `test_utils::MiniCore`) declaration at
/// the start of your tests:
///
/// ```text
/// //- minicore: iterator, ord, derive
/// ```
pub struct FamousDefs<'a, 'b>(pub &'a Semantics<'b, RootDatabase>, pub Crate);

#[allow(non_snake_case)]
impl FamousDefs<'_, '_> {
    pub fn std(&self) -> Option<Crate> {
        self.find_lang_crate(LangCrateOrigin::Std)
    }

    pub fn core(&self) -> Option<Crate> {
        self.find_lang_crate(LangCrateOrigin::Core)
    }

    pub fn alloc(&self) -> Option<Crate> {
        self.find_lang_crate(LangCrateOrigin::Alloc)
    }

    pub fn test(&self) -> Option<Crate> {
        self.find_lang_crate(LangCrateOrigin::Test)
    }

    pub fn proc_macro(&self) -> Option<Crate> {
        self.find_lang_crate(LangCrateOrigin::ProcMacro)
    }

    pub fn core_cmp_Ord(&self) -> Option<Trait> {
        self.find_trait("core:cmp:Ord")
    }

    pub fn core_convert_FromStr(&self) -> Option<Trait> {
        self.find_trait("core:str:FromStr")
    }

    pub fn core_convert_From(&self) -> Option<Trait> {
        self.find_trait("core:convert:From")
    }

    pub fn core_convert_Into(&self) -> Option<Trait> {
        self.find_trait("core:convert:Into")
    }

    pub fn core_convert_TryFrom(&self) -> Option<Trait> {
        self.find_trait("core:convert:TryFrom")
    }

    pub fn core_convert_TryInto(&self) -> Option<Trait> {
        self.find_trait("core:convert:TryInto")
    }

    pub fn core_convert_Index(&self) -> Option<Trait> {
        self.find_trait("core:ops:Index")
    }

    pub fn core_option_Option(&self) -> Option<Enum> {
        self.find_enum("core:option:Option")
    }

    pub fn core_result_Result(&self) -> Option<Enum> {
        self.find_enum("core:result:Result")
    }

    pub fn core_default_Default(&self) -> Option<Trait> {
        self.find_trait("core:default:Default")
    }

    pub fn core_iter_Iterator(&self) -> Option<Trait> {
        self.find_trait("core:iter:traits:iterator:Iterator")
    }

    pub fn core_iter_IntoIterator(&self) -> Option<Trait> {
        self.find_trait("core:iter:traits:collect:IntoIterator")
    }

    pub fn core_iter(&self) -> Option<Module> {
        self.find_module("core:iter")
    }

    pub fn core_ops_Deref(&self) -> Option<Trait> {
        self.find_trait("core:ops:Deref")
    }

    pub fn core_ops_DerefMut(&self) -> Option<Trait> {
        self.find_trait("core:ops:DerefMut")
    }

    pub fn core_convert_AsRef(&self) -> Option<Trait> {
        self.find_trait("core:convert:AsRef")
    }

    pub fn core_convert_AsMut(&self) -> Option<Trait> {
        self.find_trait("core:convert:AsMut")
    }

    pub fn core_borrow_Borrow(&self) -> Option<Trait> {
        self.find_trait("core:borrow:Borrow")
    }

    pub fn core_borrow_BorrowMut(&self) -> Option<Trait> {
        self.find_trait("core:borrow:BorrowMut")
    }

    pub fn core_ops_ControlFlow(&self) -> Option<Enum> {
        self.find_enum("core:ops:ControlFlow")
    }

    pub fn core_ops_Drop(&self) -> Option<Trait> {
        self.find_trait("core:ops:Drop")
    }

    pub fn core_marker_Copy(&self) -> Option<Trait> {
        self.find_trait("core:marker:Copy")
    }

    pub fn core_marker_Sized(&self) -> Option<Trait> {
        self.find_trait("core:marker:Sized")
    }

    pub fn core_future_Future(&self) -> Option<Trait> {
        self.find_trait("core:future:Future")
    }

    pub fn core_macros_builtin_derive(&self) -> Option<Macro> {
        self.find_macro("core:macros:builtin:derive")
    }

    pub fn core_mem_drop(&self) -> Option<Function> {
        self.find_function("core:mem:drop")
    }

    pub fn core_macros_todo(&self) -> Option<Macro> {
        self.find_macro("core:todo")
    }

    pub fn core_macros_unimplemented(&self) -> Option<Macro> {
        self.find_macro("core:unimplemented")
    }

    pub fn core_fmt_Display(&self) -> Option<Trait> {
        self.find_trait("core:fmt:Display")
    }

    pub fn alloc_string_ToString(&self) -> Option<Trait> {
        self.find_trait("alloc:string:ToString")
    }
    pub fn builtin_crates(&self) -> impl Iterator<Item = Crate> {
        IntoIterator::into_iter([
            self.std(),
            self.core(),
            self.alloc(),
            self.test(),
            self.proc_macro(),
        ])
        .flatten()
    }

    fn find_trait(&self, path: &str) -> Option<Trait> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Trait(it)) => Some(it),
            _ => None,
        }
    }

    fn find_macro(&self, path: &str) -> Option<Macro> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Macro(it)) => Some(it),
            _ => None,
        }
    }

    fn find_enum(&self, path: &str) -> Option<Enum> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Adt(hir::Adt::Enum(it))) => Some(it),
            _ => None,
        }
    }

    fn find_module(&self, path: &str) -> Option<Module> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Module(it)) => Some(it),
            _ => None,
        }
    }

    fn find_function(&self, path: &str) -> Option<Function> {
        match self.find_def(path)? {
            hir::ScopeDef::ModuleDef(hir::ModuleDef::Function(it)) => Some(it),
            _ => None,
        }
    }

    fn find_lang_crate(&self, origin: LangCrateOrigin) -> Option<Crate> {
        let krate = self.1;
        let db = self.0.db;
        let res = krate
            .dependencies(db)
            .into_iter()
            .find(|dep| dep.krate.origin(db) == CrateOrigin::Lang(origin))?
            .krate;
        Some(res)
    }

    fn find_def(&self, path: &str) -> Option<ScopeDef> {
        let db = self.0.db;
        let mut path = path.split(':');
        let trait_ = path.next_back()?;
        let lang_crate = path.next()?;
        let lang_crate = match LangCrateOrigin::from(lang_crate) {
            LangCrateOrigin::Other => return None,
            lang_crate => lang_crate,
        };
        let std_crate = self.find_lang_crate(lang_crate)?;
        let mut module = std_crate.root_module();
        for segment in path {
            module = module.children(db).find_map(|child| {
                let name = child.name(db)?;
                if name.as_str() == segment { Some(child) } else { None }
            })?;
        }
        let def =
            module.scope(db, None).into_iter().find(|(name, _def)| name.as_str() == trait_)?.1;
        Some(def)
    }
}
