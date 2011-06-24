// xfail-stage0
// error-pattern:|| cannot be applied to type `f32`

fn main() {
  auto x = 1.0_f32 || 2.0_f32;
}