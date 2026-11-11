// Note: it's ok to interpret 'a as 'a', but not ok to interpret 'abc as
// 'abc' because 'abc' is not a valid char literal.

fn main() {
    let c = 'a;
    //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
    //~| HELP add `'` to close the char literal

    let c = 'abc;
    //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
    //~| ERROR expected expression, found `;`
}

fn f() {
    match 'a' {
        'a'..='b => {}
        //~^ ERROR unexpected token: `'b`
        //~| HELP add `'` to close the char literal
        'c'..='def => {}
        //~^ ERROR unexpected token: `'def`
    }
}

fn g() {
   match 'g' {
       'g => {}
       //~^ ERROR expected pattern, found `=>`
       //~| HELP add `'` to close the char literal
       'hij => {}
       //~^ ERROR expected pattern, found `'hij`
       _ => {}
   }
}

fn h() {
   let x = ['a, 'b, 'cde];
   //~^ ERROR expected `while`, `for`, `loop` or `{` after a label
   //~| HELP add `'` to close the char literal
   //~| ERROR expected `while`, `for`, `loop` or `{` after a label
   //~| HELP add `'` to close the char literal
   //~| ERROR expected `while`, `for`, `loop` or `{` after a label
   //~| ERROR expected expression, found `]`
}
