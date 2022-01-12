// Test that AST json serialization doesn't ICE (#63728).

// revisions: expand noexpand

//[expand] compile-flags: -Zast-json
//[noexpand] compile-flags: -Zast-json-noexpand

// check-pass
// dont-check-compiler-stdout - don't check for any AST change.

enum V {
    A(i32),
    B { f: [i64; 3 + 4] }
}

trait X {
    type Output;
    fn read(&self) -> Self::Output;
    fn write(&mut self, _: Self::Output);
}

macro_rules! call_println {
    ($y:ident) => { println!("{}", $y) }
}

fn main() {
    let x: (i32) = 35;
    let y = x as i64<> + 5;

    call_println!(y);

    struct A;
}

// Regressions tests for issues #78398 and #78510 (captured tokens in associated and foreign items)

struct S;

macro_rules! mac_extern {
    ($i:item) => {
        extern "C" { $i }
    }
}
macro_rules! mac_assoc {
    ($i:item) => {
        impl S { $i }
        trait Bar { $i }
    }
}

mac_extern! {
    fn foo();
}
mac_assoc! {
    fn foo() {}
}
