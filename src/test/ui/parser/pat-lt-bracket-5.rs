// compile-flags: -Z parse-only

fn main() {
    let v[0] = v[1]; //~ ERROR expected one of `:`, `;`, `=`, or `@`, found `[`
}
