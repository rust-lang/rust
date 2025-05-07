// rustfmt-style_edition: 2024

// Based on the issue description
pub trait PrettyPrinter<'tcx>:
    Printer<
        'tcx,
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
        DynExistential = Self,
        Const = Self,
    >
{
    //
}
pub trait PrettyPrinter<'tcx>:
    Printer<
        'tcx,
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
        DynExistential = Self,
        Const = Self,
    > + fmt::Write
{
    //
}
pub trait PrettyPrinter<'tcx>:
    Printer<
        'tcx,
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
        DynExistential = Self,
        Const = Self,
    > + fmt::Write1
    + fmt::Write2
{
    //
}
pub trait PrettyPrinter<'tcx>:
    fmt::Write
    + Printer<
        'tcx,
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
        DynExistential = Self,
        Const = Self,
    >
{
    //
}
pub trait PrettyPrinter<'tcx>:
    fmt::Write
    + Printer1<
        'tcx,
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
        DynExistential = Self,
        Const = Self,
    > + Printer2<
        'tcx,
        Error = fmt::Error,
        Path = Self,
        Region = Self,
        Type = Self,
        DynExistential = Self,
        Const = Self,
    >
{
    //
}

// Some test cases to ensure other cases formatting were not changed
fn f() -> Box<
    FnMut() -> Thing<
        WithType = LongItemName,
        Error = LONGLONGLONGLONGLONGONGEvenLongerErrorNameLongerLonger,
    >,
> {
}
fn f() -> Box<
    FnMut() -> Thing<
            WithType = LongItemName,
            Error = LONGLONGLONGLONGLONGONGEvenLongerErrorNameLongerLonger,
        > + fmt::Write1
        + fmt::Write2,
> {
}

fn foo<F>(foo2: F)
where
    F: Fn(
        // this comment is deleted
    ),
{
}
fn foo<F>(foo2: F)
where
    F: Fn(
            // this comment is deleted
        ) + fmt::Write,
{
}

fn elaborate_bounds<F>(mut mk_cand: F)
where
    F: FnMut(
        &mut ProbeContext,
        ty::PolyTraitRefffffffffffffffffffffffffffffffff,
        tyyyyyyyyyyyyyyyyyyyyy::AssociatedItem,
    ),
{
}
fn elaborate_bounds<F>(mut mk_cand: F)
where
    F: FnMut(
            &mut ProbeContext,
            ty::PolyTraitRefffffffffffffffffffffffffffffffff,
            tyyyyyyyyyyyyyyyyyyyyy::AssociatedItem,
        ) + fmt::Write,
{
}

fn build_sorted_static_get_entry_names(
    mut entries: entryyyyyyyy,
) -> (
    impl Fn(
        AlphabeticalTraversal,
        Seconddddddddddddddddddddddddddddddddddd,
    ) -> Parammmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
    + Sendddddddddddddddddddddddddddddddddddddddddddd
) {
}

pub trait SomeTrait:
    Cloneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
    + Eqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
{
}

trait B = where
    for<'b> &'b Self: Send
        + Cloneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
        + Copyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy;
