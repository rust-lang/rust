fn main() {
    if false {
        if false {
        } else {
            // A let binding here seems necessary to trigger it.
            let _ = ();
        }
    } else if let false = false {
    }
}
