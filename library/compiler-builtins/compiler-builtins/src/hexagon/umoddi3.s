

FUNCTION_BEGIN __hexagon_umoddi3
 {
  r6 = cl0(r1:0)
  r7 = cl0(r3:2)
  r5:4 = r3:2
  r3:2 = r1:0
 }
 {
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
  if (p0) jump .hexagon_umoddi3_return
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

.hexagon_umoddi3_return:
 {
  r1:0 = r3:2
  jumpr r31
 }
FUNCTION_END __hexagon_umoddi3

  .globl __qdsp_umoddi3
  .set __qdsp_umoddi3, __hexagon_umoddi3
