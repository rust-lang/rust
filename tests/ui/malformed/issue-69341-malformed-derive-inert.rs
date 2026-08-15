fn main() {}

struct CLI {
    #[derive(parse())] //~ ERROR expected non-macro attribute, found attribute macro
    path: (),
}
