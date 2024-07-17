//@ check-pass

fn generate_item_fn(attr: String) {
    match attr {
        path => 'ret: {
            if false {
                break 'ret path;
            }

            return;
        }

        _ => return,
    };
}

fn main() {}
