declare fastcc i32 @rust_native_rust_1(i32, i32)

declare i32 @upcall_exit(i32)

define void @rust_new_exit_task_glue(i32, i32, i32, i32, i32) {
entry:
  %5 = inttoptr i32 %0 to void (i32, i32, i32, i32)*
  tail call fastcc void %5(i32 %1, i32 %2, i32 %3, i32 %4)
  %6 = tail call fastcc i32 @rust_native_rust_1(i32 ptrtoint (i32 (i32)* @upcall_exit to i32), i32 %2)
  ret void
}
