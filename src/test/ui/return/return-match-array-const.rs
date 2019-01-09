fn main() {
    [(); return match 0 { n => n }]; //~ ERROR: return statement outside of function body

    [(); return match 0 { 0 => 0 }]; //~ ERROR: return statement outside of function body

    [(); return match () { 'a' => 0, _ => 0 }]; //~ ERROR: return statement outside of function body
}
