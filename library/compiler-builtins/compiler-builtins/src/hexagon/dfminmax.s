 .text
 .global __hexagon_mindf3
 .global __hexagon_maxdf3
 .type __hexagon_mindf3,@function
 .type __hexagon_maxdf3,@function
 .global __qdsp_mindf3 ; .set __qdsp_mindf3, __hexagon_mindf3
 .global __qdsp_maxdf3 ; .set __qdsp_maxdf3, __hexagon_maxdf3
 .p2align 5
__hexagon_mindf3:
 {
  p0 = dfclass(r1:0,#0x10)
  p1 = dfcmp.gt(r1:0,r3:2)
  r5:4 = r1:0
 }
 {
  if (p0) r1:0 = r3:2
  if (p1) r1:0 = r3:2
  p2 = dfcmp.eq(r1:0,r3:2)
  if (!p2.new) jumpr:t r31
 }

 {
  r1:0 = or(r5:4,r3:2)
  jumpr r31
 }
.size __hexagon_mindf3,.-__hexagon_mindf3
 .falign
__hexagon_maxdf3:
 {
  p0 = dfclass(r1:0,#0x10)
  p1 = dfcmp.gt(r3:2,r1:0)
  r5:4 = r1:0
 }
 {
  if (p0) r1:0 = r3:2
  if (p1) r1:0 = r3:2
  p2 = dfcmp.eq(r1:0,r3:2)
  if (!p2.new) jumpr:t r31
 }

 {
  r1:0 = and(r5:4,r3:2)
  jumpr r31
 }
.size __hexagon_maxdf3,.-__hexagon_maxdf3
