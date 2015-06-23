
fn simple(// pre-comment on a function!?
          i: i32, // yes, it's possible!
          response: NoWay /* hose */) {
    "cool"
}


fn weird_comment(// /*/ double level */ comment
                 x: Hello, // /*/* tripple, even */*/
                 // Does this work?
                 y: World) {
    simple(// does this preserve comments now?
           42,
           NoWay)
}
