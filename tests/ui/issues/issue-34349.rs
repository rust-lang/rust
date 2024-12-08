// This is a regression test for a problem encountered around upvar
// inference and trait caching: in particular, we were entering a
// temporary closure kind during inference, and then caching results
// based on that temporary kind, which led to no error being reported
// in this particular test.

fn main() {
    let inc = || {};
    inc();

    fn apply<F>(f: F) where F: Fn() {
        f()
    }

    let mut farewell = "goodbye".to_owned();
    let diary = || { //~ ERROR E0525
        farewell.push_str("!!!");
        println!("Then I screamed {}.", farewell);
    };

    apply(diary);
}
