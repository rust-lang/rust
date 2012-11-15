// xfail-test
const generations: uint = 1024+256+128+49;

fn child_no(x: uint) -> fn~() {
     || {
        if x < generations {
            task::spawn(child_no(x+1));
        }
    }
}

fn main() {
    task::spawn(child_no(0));
}
