//@ check-pass
// Test cases involving principal-less traits (dyn Send without a primary trait).

fn lifetime_cast_send<'a, 'b>(a: *mut (dyn Send + 'a)) -> *mut (dyn Send + 'b) {
    a as _
}

fn main() {}
