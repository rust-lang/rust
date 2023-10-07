fn main() {
    let x = 1;
    let y = 2;
    let value = 3;

    match value {
        Some(x) if x == y {
            self.next_token()?; //~ ERROR expected identifier, found keyword `self`
        },
        _ => {}
    }
    let _: i32 = (); //~ ERROR mismatched types
}

struct Foo {
    value: usize
}

fn foo(a: Option<&mut Foo>, b: usize) {
    match a {
        Some(a) if a.value == b {
            a.value = 1; //~ ERROR expected one of `,`, `:`, or `}`, found `.`
        },
        _ => {}
    }
    let _: i32 = (); //~ ERROR mismatched types
}

fn bar(a: Option<&mut Foo>, b: usize) {
    match a {
        Some(a) if a.value == b {
            a.value, //~ ERROR expected one of `,`, `:`, or `}`, found `.`
        } => {
        }
        _ => {}
    }
    let _: i32 = (); //~ ERROR mismatched types
}
