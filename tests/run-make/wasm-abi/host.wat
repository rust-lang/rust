(module
  (func (export "two_i32") (result i32 i32)
      i32.const 100
      i32.const 101)
  (func (export "two_i64") (result i64 i64)
      i64.const 102
      i64.const 103)
  (func (export "two_f32") (result f32 f32)
      f32.const 104
      f32.const 105)
  (func (export "two_f64") (result f64 f64)
      f64.const 106
      f64.const 107)

  (func (export "mishmash") (result f64 f32 i32 i64 i32 i32)
      f64.const 108
      f32.const 109
      i32.const 110
      i64.const 111
      i32.const 112
      i32.const 113)
)
