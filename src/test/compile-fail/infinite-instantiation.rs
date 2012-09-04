// error-pattern: overly deep expansion
// issue 2258

trait to_opt {
    fn to_option() -> Option<self>;
}

impl uint: to_opt {
    fn to_option() -> Option<uint> {
        Some(self)
    }
}

impl<T:copy> Option<T>: to_opt {
    fn to_option() -> Option<Option<T>> {
        Some(self)
    }
}

fn function<T:to_opt>(counter: uint, t: T) {
    if counter > 0u {
        function(counter - 1u, t.to_option());
    }
}

fn main() {
    function(22u, 22u);
}
