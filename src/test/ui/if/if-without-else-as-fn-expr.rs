fn foo(bar: usize) -> usize {
    if bar % 5 == 0 {
        return 3;
    }
    //~^^^ ERROR if may be missing an else clause
    //~| ERROR mismatched types [E0308]
}

fn foo2(bar: usize) -> usize {
    let x: usize = if bar % 5 == 0 {
        return 3;
    };
    //~^^^ ERROR if may be missing an else clause
    //~| ERROR mismatched types [E0308]
    x
}

fn foo3(bar: usize) -> usize {
    if bar % 5 == 0 {
        3
    }
    //~^^^ ERROR if may be missing an else clause
    //~| ERROR mismatched types [E0308]
}

fn foo_let(bar: usize) -> usize {
    if let 0 = 1 {
        return 3;
    }
    //~^^^ ERROR if may be missing an else clause
    //~| ERROR mismatched types [E0308]
}

fn foo2_let(bar: usize) -> usize {
    let x: usize = if let 0 = 1 {
        return 3;
    };
    //~^^^ ERROR if may be missing an else clause
    //~| ERROR mismatched types [E0308]
    x
}

fn foo3_let(bar: usize) -> usize {
    if let 0 = 1 {
        3
    }
    //~^^^ ERROR if may be missing an else clause
    //~| ERROR mismatched types [E0308]
}

// FIXME(60254): deduplicate first error in favor of second.

fn main() {
    let _ = foo(1);
}
