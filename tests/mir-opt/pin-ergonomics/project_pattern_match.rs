// skip-filecheck
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This test verifies that a `&pin mut Foo` can be projected to a pinned
// reference `&pin mut T` of a `?Unpin` field , and can be projected to
// an unpinned reference `&mut U` of an `Unpin` field .

struct Foo<T, U> {
    x: T,
    y: U,
}

struct Bar<T, U>(T, U);

enum Baz<T, U> {
    Foo(T, U),
    Bar { x: T, y: U },
}

// EMIT_MIR project_pattern_match.foo_mut.built.after.mir
fn foo_mut<T, U: Unpin>(foo: &pin mut Foo<T, U>) {
    let Foo { x, y } = foo;
}

// EMIT_MIR project_pattern_match.foo_const.built.after.mir
fn foo_const<T, U: Unpin>(foo: &pin const Foo<T, U>) {
    let Foo { x, y } = foo;
}

// EMIT_MIR project_pattern_match.bar_mut.built.after.mir
fn bar_mut<T, U: Unpin>(bar: &pin mut Bar<T, U>) {
    let Bar(x, y) = bar;
}

// EMIT_MIR project_pattern_match.bar_const.built.after.mir
fn bar_const<T, U: Unpin>(bar: &pin const Bar<T, U>) {
    let Bar(x, y) = bar;
}

// EMIT_MIR project_pattern_match.foo_bar_mut.built.after.mir
fn foo_bar_mut<T, U: Unpin>(foo: &pin mut Foo<Bar<T, U>, Bar<T, U>>) {
    let Foo { x: Bar(x, y), y: Bar(z, w) } = foo;
}

// EMIT_MIR project_pattern_match.foo_bar_const.built.after.mir
fn foo_bar_const<T, U: Unpin>(foo: &pin const Foo<Bar<T, U>, Bar<T, U>>) {
    let Foo { x: Bar(x, y), y: Bar(z, w) } = foo;
}

// EMIT_MIR project_pattern_match.baz_mut.built.after.mir
fn baz_mut<T, U: Unpin>(baz: &pin mut Baz<T, U>) {
    match baz {
        Baz::Foo(x, y) => {}
        Baz::Bar { x, y } => {}
    }
}

// EMIT_MIR project_pattern_match.baz_const.built.after.mir
fn baz_const<T, U: Unpin>(baz: &pin const Baz<T, U>) {
    match baz {
        Baz::Foo(x, y) => {}
        Baz::Bar { x, y } => {}
    }
}

// EMIT_MIR project_pattern_match.baz_baz_mut.built.after.mir
fn baz_baz_mut<T, U: Unpin>(baz: &pin mut Baz<Baz<T, U>, Baz<T, U>>) {
    match baz {
        Baz::Foo(Baz::Foo(x, y), Baz::Foo(z, w)) => {}
        Baz::Foo(Baz::Foo(x, y), Baz::Bar { x: z, y: w }) => {}
        Baz::Foo(Baz::Bar { x, y }, Baz::Foo(z, w)) => {}
        Baz::Foo(Baz::Bar { x, y }, Baz::Bar { x: z, y: w }) => {}
        Baz::Bar { x: Baz::Foo(x, y), y: Baz::Foo(z, w) } => {}
        Baz::Bar { x: Baz::Foo(x, y), y: Baz::Bar { x: z, y: w } } => {}
        Baz::Bar { x: Baz::Bar { x, y }, y: Baz::Foo(z, w) } => {}
        Baz::Bar { x: Baz::Bar { x, y }, y: Baz::Bar { x: z, y: w } } => {}
    }
}

// EMIT_MIR project_pattern_match.baz_baz_const.built.after.mir
fn baz_baz_const<T, U: Unpin>(baz: &pin const Baz<Baz<T, U>, Baz<T, U>>) {
    match baz {
        Baz::Foo(Baz::Foo(x, y), Baz::Foo(z, w)) => {}
        Baz::Foo(Baz::Foo(x, y), Baz::Bar { x: z, y: w }) => {}
        Baz::Foo(Baz::Bar { x, y }, Baz::Foo(z, w)) => {}
        Baz::Foo(Baz::Bar { x, y }, Baz::Bar { x: z, y: w }) => {}
        Baz::Bar { x: Baz::Foo(x, y), y: Baz::Foo(z, w) } => {}
        Baz::Bar { x: Baz::Foo(x, y), y: Baz::Bar { x: z, y: w } } => {}
        Baz::Bar { x: Baz::Bar { x, y }, y: Baz::Foo(z, w) } => {}
        Baz::Bar { x: Baz::Bar { x, y }, y: Baz::Bar { x: z, y: w } } => {}
    }
}

fn main() {}
