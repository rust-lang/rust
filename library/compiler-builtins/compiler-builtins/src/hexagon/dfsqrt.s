 .text
 .global __hexagon_sqrtdf2
 .type __hexagon_sqrtdf2,@function
 .global __hexagon_sqrt
 .type __hexagon_sqrt,@function
 .global __qdsp_sqrtdf2 ; .set __qdsp_sqrtdf2, __hexagon_sqrtdf2; .type __qdsp_sqrtdf2,@function
 .global __qdsp_sqrt ; .set __qdsp_sqrt, __hexagon_sqrt; .type __qdsp_sqrt,@function
 .global __hexagon_fast_sqrtdf2 ; .set __hexagon_fast_sqrtdf2, __hexagon_sqrtdf2; .type __hexagon_fast_sqrtdf2,@function
 .global __hexagon_fast_sqrt ; .set __hexagon_fast_sqrt, __hexagon_sqrt; .type __hexagon_fast_sqrt,@function
 .global __hexagon_fast2_sqrtdf2 ; .set __hexagon_fast2_sqrtdf2, __hexagon_sqrtdf2; .type __hexagon_fast2_sqrtdf2,@function
 .global __hexagon_fast2_sqrt ; .set __hexagon_fast2_sqrt, __hexagon_sqrt; .type __hexagon_fast2_sqrt,@function
 .type sqrt,@function
 .p2align 5
__hexagon_sqrtdf2:
__hexagon_sqrt:
 {
  r15:14 = extractu(r1:0,#23 +1,#52 -23)
  r28 = extractu(r1,#11,#52 -32)
  r5:4 = combine(##0x3f000004,#1)
 }
 {
  p2 = dfclass(r1:0,#0x02)
  p2 = cmp.gt(r1,#-1)
  if (!p2.new) jump:nt .Lsqrt_abnormal
  r9 = or(r5,r14)
 }

.Ldenormal_restart:
 {
  r11:10 = r1:0
  r7,p0 = sfinvsqrta(r9)
  r5 = and(r5,#-16)
  r3:2 = #0
 }
 {
  r3 += sfmpy(r7,r9):lib
  r2 += sfmpy(r7,r5):lib
  r6 = r5


  r9 = and(r28,#1)
 }
 {
  r6 -= sfmpy(r3,r2):lib
  r11 = insert(r4,#11 +1,#52 -32)
  p1 = cmp.gtu(r9,#0)
 }
 {
  r3 += sfmpy(r3,r6):lib
  r2 += sfmpy(r2,r6):lib
  r6 = r5
  r9 = mux(p1,#8,#9)
 }
 {
  r6 -= sfmpy(r3,r2):lib
  r11:10 = asl(r11:10,r9)
  r9 = mux(p1,#3,#2)
 }
 {
  r2 += sfmpy(r2,r6):lib

  r15:14 = asl(r11:10,r9)
 }
 {
  r2 = and(r2,##0x007fffff)
 }
 {
  r2 = add(r2,##0x00800000 - 3)
  r9 = mux(p1,#7,#8)
 }
 {
  r8 = asl(r2,r9)
  r9 = mux(p1,#15-(1+1),#15-(1+0))
 }
 {
  r13:12 = mpyu(r8,r15)
 }
 {
  r1:0 = asl(r11:10,#15)
  r15:14 = mpyu(r13,r13)
  p1 = cmp.eq(r0,r0)
 }
 {
  r1:0 -= asl(r15:14,#15)
  r15:14 = mpyu(r13,r12)
  p2 = cmp.eq(r0,r0)
 }
 {
  r1:0 -= lsr(r15:14,#16)
  p3 = cmp.eq(r0,r0)
 }
 {
  r1:0 = mpyu(r1,r8)
 }
 {
  r13:12 += lsr(r1:0,r9)
  r9 = add(r9,#16)
  r1:0 = asl(r11:10,#31)
 }

 {
  r15:14 = mpyu(r13,r13)
  r1:0 -= mpyu(r13,r12)
 }
 {
  r1:0 -= asl(r15:14,#31)
  r15:14 = mpyu(r12,r12)
 }
 {
  r1:0 -= lsr(r15:14,#33)
 }
 {
  r1:0 = mpyu(r1,r8)
 }
 {
  r13:12 += lsr(r1:0,r9)
  r9 = add(r9,#16)
  r1:0 = asl(r11:10,#47)
 }

 {
  r15:14 = mpyu(r13,r13)
 }
 {
  r1:0 -= asl(r15:14,#47)
  r15:14 = mpyu(r13,r12)
 }
 {
  r1:0 -= asl(r15:14,#16)
  r15:14 = mpyu(r12,r12)
 }
 {
  r1:0 -= lsr(r15:14,#17)
 }
 {
  r1:0 = mpyu(r1,r8)
 }
 {
  r13:12 += lsr(r1:0,r9)
 }
 {
  r3:2 = mpyu(r13,r12)
  r5:4 = mpyu(r12,r12)
  r15:14 = #0
  r1:0 = #0
 }
 {
  r3:2 += lsr(r5:4,#33)
  r5:4 += asl(r3:2,#33)
  p1 = cmp.eq(r0,r0)
 }
 {
  r7:6 = mpyu(r13,r13)
  r1:0 = sub(r1:0,r5:4,p1):carry
  r9:8 = #1
 }
 {
  r7:6 += lsr(r3:2,#31)
  r9:8 += asl(r13:12,#1)
 }





 {
  r15:14 = sub(r11:10,r7:6,p1):carry
  r5:4 = sub(r1:0,r9:8,p2):carry




  r7:6 = #1
  r11:10 = #0
 }
 {
  r3:2 = sub(r15:14,r11:10,p2):carry
  r7:6 = add(r13:12,r7:6)
  r28 = add(r28,#-0x3ff)
 }
 {

  if (p2) r13:12 = r7:6
  if (p2) r1:0 = r5:4
  if (p2) r15:14 = r3:2
 }
 {
  r5:4 = sub(r1:0,r9:8,p3):carry
  r7:6 = #1
  r28 = asr(r28,#1)
 }
 {
  r3:2 = sub(r15:14,r11:10,p3):carry
  r7:6 = add(r13:12,r7:6)
 }
 {
  if (p3) r13:12 = r7:6
  if (p3) r1:0 = r5:4





  r2 = #1
 }
 {
  p0 = cmp.eq(r1:0,r11:10)
  if (!p0.new) r12 = or(r12,r2)
  r3 = cl0(r13:12)
  r28 = add(r28,#-63)
 }



 {
  r1:0 = convert_ud2df(r13:12)
  r28 = add(r28,r3)
 }
 {
  r1 += asl(r28,#52 -32)
  jumpr r31
 }
.Lsqrt_abnormal:
 {
  p0 = dfclass(r1:0,#0x01)
  if (p0.new) jumpr:t r31
 }
 {
  p0 = dfclass(r1:0,#0x10)
  if (p0.new) jump:nt .Lsqrt_nan
 }
 {
  p0 = cmp.gt(r1,#-1)
  if (!p0.new) jump:nt .Lsqrt_invalid_neg
  if (!p0.new) r28 = ##0x7F800001
 }
 {
  p0 = dfclass(r1:0,#0x08)
  if (p0.new) jumpr:nt r31
 }


 {
  r1:0 = extractu(r1:0,#52,#0)
 }
 {
  r28 = add(clb(r1:0),#-11)
 }
 {
  r1:0 = asl(r1:0,r28)
  r28 = sub(#1,r28)
 }
 {
  r1 = insert(r28,#1,#52 -32)
 }
 {
  r3:2 = extractu(r1:0,#23 +1,#52 -23)
  r5 = ##0x3f000004
 }
 {
  r9 = or(r5,r2)
  r5 = and(r5,#-16)
  jump .Ldenormal_restart
 }
.Lsqrt_nan:
 {
  r28 = convert_df2sf(r1:0)
  r1:0 = #-1
  jumpr r31
 }
.Lsqrt_invalid_neg:
 {
  r1:0 = convert_sf2df(r28)
  jumpr r31
 }
.size __hexagon_sqrt,.-__hexagon_sqrt
.size __hexagon_sqrtdf2,.-__hexagon_sqrtdf2
