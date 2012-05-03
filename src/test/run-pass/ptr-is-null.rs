fn main() {
   let p: *int = ptr::null();
   assert p.is_null();
   assert !p.is_not_null();

   let q = ptr::offset(p, 1u);
   assert !q.is_null();
   assert q.is_not_null();
}
