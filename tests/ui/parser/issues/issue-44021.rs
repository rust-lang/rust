struct MyStruct;
impl MyStruct {
    fn f() {|x, y} //~ ERROR expected one of `:`, `@`, or `|`, found `}`
}

fn main() {}
