// Verify that we do not ICE when failing to parse a statement in `cfg_eval`.

#![feature(cfg_eval)]
#![feature(stmt_expr_attributes)]

#[cfg_eval]
fn main() {
    #[cfg_eval]
    let _ = #[cfg(false)] 0;
    //~^ ERROR removing an expression is not supported in this position
    //~| ERROR expected expression, found `;`
    //~| ERROR removing an expression is not supported in this position
}
