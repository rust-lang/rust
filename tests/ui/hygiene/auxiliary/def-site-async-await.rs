//@ edition:2018
//@ proc-macro: opaque-hygiene.rs

pub async fn serve() {
    opaque_hygiene::make_it!();
}
