//@ check-pass

#[cfg(false)]
fn check() {
    #[attr] x = y; //~ WARN ambiguous expression attributes are unstable
                   //~| WARN unstable syntax can change at any point in the future
    x = #[attr] y = z; //~ WARN ambiguous expression attributes are unstable
                       //~| WARN unstable syntax can change at any point in the future
    #[attr] val as Type; //~ WARN ambiguous expression attributes are unstable
                         //~| WARN unstable syntax can change at any point in the future
}

fn main() {}
