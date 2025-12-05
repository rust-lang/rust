fn foo() {
   for _x in 0 .. (0 .. {1 + 2}).sum::<u32>() {
       break;
   }
}
