// compile-flags: -Z parse-only

fn main() {
    let buf[0] = 0; //~ ERROR expected one of `:`, `;`, `=`, or `@`, found `[`
}
