fn wants_static_fn(_x: &static/fn()) {}

fn main() {
    let i = 3;
    do wants_static_fn { //~ ERROR cannot infer an appropriate lifetime due to conflicting requirements
        debug!("i=%d", i);
    }
}
