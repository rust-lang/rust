// The .stderr file will contain the color highlighting in `mismatched types`
// errors for function types and function pointers.
//
// If you want to view the .stderr file after changes occurred, there are extensions
// for IDEs -- e.g. "ANSI Colors" for vscode -- to view ANSI-colored text properly.
// Of course, alternatively, you can just view the file
// `cat src/test/ui/function-type-mismatch-color-highlights.stderr`
// and/or the `git diff` in a terminal that can render the colors for you.

// compile-flags: --json=diagnostic-rendered-ansi

fn main() {
    fn f() {}
    fn g() {}
    fn g2() -> u8 {
        0
    }
    fn g3(_: u8) {}

    fn h<T>() {}
    fn h2<T>(_: T) {}

    struct Struct();

    let mut x = f;
    x = || (); //~ ERROR mismatched types [E0308]
    x = (|| ()) as fn(); //~ ERROR mismatched types [E0308]
    x = (|| 0) as fn() -> u8; //~ ERROR mismatched types [E0308]
    x = (|_| ()) as fn(u8); //~ ERROR mismatched types [E0308]
    x = g; //~ ERROR mismatched types [E0308]
    x = g2; //~ ERROR mismatched types [E0308]
    x = g3; //~ ERROR mismatched types [E0308]
    x = Struct; //~ ERROR mismatched types [E0308]

    let mut y = h::<()>;
    y = h::<u8>; //~ ERROR mismatched types [E0308]

    let mut y2 = h2::<()>;
    y = h2::<u8>; //~ ERROR mismatched types [E0308]

    let mut z = g3;
    z = (|| ()) as fn(); //~ ERROR mismatched types [E0308]

    let mut p = (|| ()) as fn();
    p = (|| 0) as fn() -> u8; //~ ERROR mismatched types [E0308]
    p = (|_| ()) as fn(u8); //~ ERROR mismatched types [E0308]
    let _: fn() -> u8 = g3; //~ ERROR mismatched types [E0308]
    let _: fn() -> u16 = g2; //~ ERROR mismatched types [E0308]
}
