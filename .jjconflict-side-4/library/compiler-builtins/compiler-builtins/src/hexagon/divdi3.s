
FUNCTION_BEGIN __hexagon_divdi3
 {
  p2 = tstbit(r1,#31)
  p3 = tstbit(r3,#31)
 }
 {
  r1:0 = abs(r1:0)
  r3:2 = abs(r3:2)
 }
 {
  r6 = cl0(r1:0)
  r7 = cl0(r3:2)
  r5:4 = r3:2
  r3:2 = r1:0
 }
 {
  p3 = xor(p2,p3)
  r10 = sub(r7,r6)
  r1:0 = #0
  r15:14 = #1
 }
 {
  r11 = add(r10,#1)
  r13:12 = lsl(r5:4,r10)
  r15:14 = lsl(r15:14,r10)
 }
 {
  p0 = cmp.gtu(r5:4,r3:2)
  loop0(1f,r11)
 }
 {
  if (p0) jump .hexagon_divdi3_return
 }
 .falign
1:
 {
  p0 = cmp.gtu(r13:12,r3:2)
 }
 {
  r7:6 = sub(r3:2, r13:12)
  r9:8 = add(r1:0, r15:14)
 }
 {
  r1:0 = vmux(p0, r1:0, r9:8)
  r3:2 = vmux(p0, r3:2, r7:6)
 }
 {
  r15:14 = lsr(r15:14, #1)
  r13:12 = lsr(r13:12, #1)
 }:endloop0

.hexagon_divdi3_return:
 {
  r3:2 = neg(r1:0)
 }
 {
  r1:0 = vmux(p3,r3:2,r1:0)
  jumpr r31
 }
FUNCTION_END __hexagon_divdi3

  .globl __qdsp_divdi3
  .set __qdsp_divdi3, __hexagon_divdi3
