// compile-flags: -Z parse-only

fn main() {
    match 0 {
        (.. pat) => {} //~ ERROR expected one of `)` or `,`, found `pat`
    }
}
