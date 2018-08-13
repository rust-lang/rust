pub trait WriteMessage {
    fn write_message(&FrontendMessage);
}

trait Runnable {
    fn handler();
}

trait TraitWithExpr {
    fn fn_with_expr(x: [i32; 1]);
}
