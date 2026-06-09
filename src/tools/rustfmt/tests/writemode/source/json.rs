// rustfmt-fn_single_line: true
// rustfmt-emit_mode: json
// Test single-line functions.

fn foo_expr() {
    1
}

fn foo_stmt() {
    foo();
}

fn foo_decl_local()  {
    let z = 5;
   }

fn    foo_decl_item(x: &mut i32) {
    x = 3;
}

   fn empty()     {

}

fn foo_return() -> String {
    "yay"
}

fn foo_where() -> T where T: Sync {
    let x = 2;
}

fn fooblock() {
    {
        "inner-block"
    }
}

fn fooblock2(x: i32) {
    let z = match x {
        _ => 2,
    };
}

fn comment() {
    // this is a test comment
    1
}

fn comment2() {
    // multi-line comment
    let z = 2;
    1
}

fn only_comment() {
    // Keep this here
}

fn aaaaaaaaaaaaaaaaa_looooooooooooooooooooooong_name() {
    let z = "aaaaaaawwwwwwwwwwwwwwwwwwwwwwwwwwww";
}

fn lots_of_space                      ()                                                           {
                           1                 
}

fn mac() -> Vec<i32> { vec![] }

trait CoolTypes {
    fn dummy(&self) {
    }
}

trait CoolerTypes { fn dummy(&self) { 
}
}

fn Foo<T>() where T: Bar {
}
