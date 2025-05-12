fn foo()
where
   'a: 'b + 'c,
   T: Clone + Copy + 'static,
   Iterator::Item: 'a,
   <T as Iterator>::Item: 'a
{}
