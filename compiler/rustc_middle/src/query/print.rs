use crate::mir::interpret::GlobalId;
use crate::query::IntoQueryParam;
use crate::query::TyCtxt;
use crate::ty;
use rustc_hir::def_id::CrateNum;
use rustc_hir::def_id::DefId;
use rustc_span::Symbol;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;

/// This is used to print query keys without giving them access to `TyCtxt`.
/// This enables more reliable printing when printing the query stack on panics.
#[derive(Copy, Clone)]
pub struct PrintContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    use_queries: bool,
}

impl<'tcx> PrintContext<'tcx> {
    pub fn arg<T>(&self, arg: T) -> PrintArg<'tcx, T> {
        PrintArg { arg: Some(arg), ctx: *self }
    }

    pub fn def_path_str(&self, def_id: PrintArg<'tcx, impl IntoQueryParam<DefId>>) -> String {
        def_id.print(|arg| {
            let def_id = arg.into_query_param();
            if self.use_queries {
                self.tcx.def_path_str(def_id)
            } else {
                self.tcx.safe_def_path_str_untracked(def_id)
            }
        })
    }
}

/// This runs some code in a printing context. If `use_queries` is false this function should
/// ensure that no queries are run.
pub fn run<'tcx, R>(
    tcx: TyCtxt<'tcx>,
    use_queries: bool,
    print: impl FnOnce(PrintContext<'tcx>) -> R,
) -> R {
    let ctx = PrintContext { tcx, use_queries };
    if use_queries { ty::print::with_no_trimmed_paths!(print(ctx)) } else { print(ctx) }
}

const INACCESSIBLE: &'static str = "<inaccessible>";

/// This wraps a value that we want to print without giving access to the regular types
/// and their Display and Debug impls. `map` and `map_with_tcx` gives access to the inner
/// type, but erases the value for the no query path.
#[derive(Copy, Clone)]
pub struct PrintArg<'tcx, T> {
    arg: Option<T>,
    ctx: PrintContext<'tcx>,
}

impl<'tcx, T> PrintArg<'tcx, T> {
    fn print(self, f: impl FnOnce(T) -> String) -> String {
        self.arg.map(f).unwrap_or_else(|| INACCESSIBLE.to_owned())
    }

    fn fmt_map(
        &self,
        f: &mut fmt::Formatter<'_>,
        map: impl FnOnce(&mut fmt::Formatter<'_>, &T) -> fmt::Result,
    ) -> fmt::Result {
        match &self.arg {
            Some(arg) => map(f, arg),
            _ => write!(f, "{}", INACCESSIBLE),
        }
    }

    fn fmt_with_queries(
        &self,
        f: &mut fmt::Formatter<'_>,
        map: impl FnOnce(&mut fmt::Formatter<'_>, &T) -> fmt::Result,
    ) -> fmt::Result {
        match &self.arg {
            Some(arg) if self.ctx.use_queries => map(f, arg),
            _ => write!(f, "{}", INACCESSIBLE),
        }
    }

    pub fn ctx(&self) -> PrintContext<'tcx> {
        self.ctx
    }

    /// Maps the argument where `f` is known to not call queries.
    fn map_unchecked<R>(self, f: impl FnOnce(T) -> R) -> PrintArg<'tcx, R> {
        PrintArg { arg: self.arg.map(f), ctx: self.ctx }
    }

    pub fn map<R>(self, f: impl FnOnce(T) -> R) -> PrintArg<'tcx, R> {
        self.map_with_tcx(|_, arg| f(arg))
    }

    pub fn map_with_tcx<R>(self, f: impl FnOnce(TyCtxt<'tcx>, T) -> R) -> PrintArg<'tcx, R> {
        PrintArg {
            arg: if self.ctx.use_queries { self.arg.map(|arg| f(self.ctx.tcx, arg)) } else { None },
            ctx: self.ctx,
        }
    }

    pub fn describe_as_module(&self) -> String
    where
        T: IntoQueryParam<DefId> + Copy,
    {
        self.print(|arg| {
            let def_id: DefId = arg.into_query_param();
            if def_id.is_top_level_module() {
                "top-level module".to_string()
            } else {
                format!("module `{}`", self.ctx.def_path_str(*self))
            }
        })
    }
}

impl<'tcx, T0, T1> PrintArg<'tcx, (T0, T1)> {
    pub fn i_0(self) -> PrintArg<'tcx, T0> {
        self.map_unchecked(|arg| arg.0)
    }
    pub fn i_1(self) -> PrintArg<'tcx, T1> {
        self.map_unchecked(|arg| arg.1)
    }
}

impl<'tcx, T> PrintArg<'tcx, ty::ParamEnvAnd<'tcx, T>> {
    pub fn value(self) -> PrintArg<'tcx, T> {
        self.map_unchecked(|arg| arg.value)
    }
}

impl<'tcx> PrintArg<'tcx, GlobalId<'tcx>> {
    pub fn display(self) -> String {
        self.print(|arg| {
            if self.ctx.use_queries {
                arg.display(self.ctx.tcx)
            } else {
                let instance_name = self.ctx.def_path_str(self.ctx.arg(arg.instance.def.def_id()));
                if let Some(promoted) = arg.promoted {
                    format!("{instance_name}::{promoted:?}")
                } else {
                    instance_name
                }
            }
        })
    }
}

impl<T: Debug> Debug for PrintArg<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_queries(f, |f, arg| arg.fmt(f))
    }
}

impl<T: Display> Display for PrintArg<'_, T> {
    default fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_queries(f, |f, arg| arg.fmt(f))
    }
}

impl Display for PrintArg<'_, CrateNum> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_map(f, |f, arg| {
            if self.ctx.use_queries {
                write!(f, "`{}`", arg)
            } else {
                write!(f, "`{}`", self.ctx.tcx.safe_crate_str_untracked(*arg))
            }
        })
    }
}

impl Display for PrintArg<'_, Symbol> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_map(f, |f, arg| fmt::Display::fmt(arg, f))
    }
}
