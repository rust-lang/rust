// CHECK: S_OBJNAME{{.*}}hotpatch_pdb{{.*}}.o
// CHECK: S_COMPILE3
// CHECK-NOT: S_
// CHECK: flags = {{.*}}hot patchable

pub fn main() {}
