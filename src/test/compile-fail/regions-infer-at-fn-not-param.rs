struct param1 {
    g: &fn()
}

struct param2 {
    g: fn()
}

struct not_param1 {
    g: @fn()
}

struct not_param2 {
    g: @fn()
}

fn take1(p: param1) -> param1 { p } //~ ERROR mismatched types
fn take2(p: param2) -> param2 { p } //~ ERROR mismatched types
fn take3(p: not_param1) -> not_param1 { p }
fn take4(p: not_param2) -> not_param2 { p }

fn main() {}
