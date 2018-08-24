struct parameterized1<'a> {
    g: Box<FnMut() + 'a>
}

struct not_parameterized1 {
    g: Box<FnMut() + 'static>
}

struct not_parameterized2 {
    g: Box<FnMut() + 'static>
}

fn take1<'a>(p: parameterized1) -> parameterized1<'a> { p }
//~^ ERROR explicit lifetime required in the type of `p`

fn take3(p: not_parameterized1) -> not_parameterized1 { p }
fn take4(p: not_parameterized2) -> not_parameterized2 { p }

fn main() {}
