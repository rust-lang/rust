 .text
 .global __hexagon_muldf3
 .type __hexagon_muldf3,@function
 .global __qdsp_muldf3 ; .set __qdsp_muldf3, __hexagon_muldf3
  .global __hexagon_fast_muldf3 ; .set __hexagon_fast_muldf3, __hexagon_muldf3
  .global __hexagon_fast2_muldf3 ; .set __hexagon_fast2_muldf3, __hexagon_muldf3
 .p2align 5
__hexagon_muldf3:
 {
  p0 = dfclass(r1:0,#2)
  p0 = dfclass(r3:2,#2)
  r13:12 = combine(##0x40000000,#0)
 }
 {
  r13:12 = insert(r1:0,#52,#11 -1)
  r5:4 = asl(r3:2,#11 -1)
  r28 = #-1024
  r9:8 = #1
 }
 {
  r7:6 = mpyu(r4,r13)
  r5:4 = insert(r9:8,#2,#62)
 }




 {
  r15:14 = mpyu(r12,r4)
  r7:6 += mpyu(r12,r5)
 }
 {
  r7:6 += lsr(r15:14,#32)
  r11:10 = mpyu(r13,r5)
  r5:4 = combine(##1024 +1024 -4,#0)
 }
 {
  r11:10 += lsr(r7:6,#32)
  if (!p0) jump .Lmul_abnormal
  p1 = cmp.eq(r14,#0)
  p1 = cmp.eq(r6,#0)
 }
 {
  if (!p1) r10 = or(r10,r8)
  r6 = extractu(r1,#11,#20)
  r7 = extractu(r3,#11,#20)
 }
 {
  r15:14 = neg(r11:10)
  r6 += add(r28,r7)
  r28 = xor(r1,r3)
 }
 {
  if (!p2.new) r11:10 = r15:14
  p2 = cmp.gt(r28,#-1)
  p0 = !cmp.gt(r6,r5)
  p0 = cmp.gt(r6,r4)
  if (!p0.new) jump:nt .Lmul_ovf_unf
 }
 {
  r1:0 = convert_d2df(r11:10)
  r6 = add(r6,#-1024 -58)
 }
 {
  r1 += asl(r6,#20)
  jumpr r31
 }

 .falign
.Lpossible_unf1:
 {
  p0 = cmp.eq(r0,#0)
  p0 = bitsclr(r1,r4)
  if (!p0.new) jumpr:t r31
  r5 = #0x7fff
 }
 {
  p0 = bitsset(r13,r5)
  r4 = USR
  r5 = #0x030
 }
 {
  if (p0) r4 = or(r4,r5)
 }
 {
  USR = r4
 }
 {
  p0 = dfcmp.eq(r1:0,r1:0)
  jumpr r31
 }
 .falign
.Lmul_ovf_unf:
 {
  r1:0 = convert_d2df(r11:10)
  r13:12 = abs(r11:10)
  r7 = add(r6,#-1024 -58)
 }
 {
  r1 += asl(r7,#20)
  r7 = extractu(r1,#11,#20)
  r4 = ##0x7FEFFFFF
 }
 {
  r7 += add(r6,##-1024 -58)

  r5 = #0
 }
 {
  p0 = cmp.gt(r7,##1024 +1024 -2)
  if (p0.new) jump:nt .Lmul_ovf
 }
 {
  p0 = cmp.gt(r7,#0)
  if (p0.new) jump:nt .Lpossible_unf1
  r5 = sub(r6,r5)
  r28 = #63
 }
 {
  r4 = #0
  r5 = sub(#5,r5)
 }
 {
  p3 = cmp.gt(r11,#-1)
  r5 = min(r5,r28)
  r11:10 = r13:12
 }
 {
  r28 = USR
  r15:14 = extractu(r11:10,r5:4)
 }
 {
  r11:10 = asr(r11:10,r5)
  r4 = #0x0030
  r1 = insert(r9,#11,#20)
 }
 {
  p0 = cmp.gtu(r9:8,r15:14)
  if (!p0.new) r10 = or(r10,r8)
  r11 = setbit(r11,#20 +3)
 }
 {
  r15:14 = neg(r11:10)
  p1 = bitsclr(r10,#0x7)
  if (!p1.new) r28 = or(r4,r28)
 }
 {
  if (!p3) r11:10 = r15:14
  USR = r28
 }
 {
  r1:0 = convert_d2df(r11:10)
  p0 = dfcmp.eq(r1:0,r1:0)
 }
 {
  r1 = insert(r9,#11 -1,#20 +1)
  jumpr r31
 }
 .falign
.Lmul_ovf:

 {
  r28 = USR
  r13:12 = combine(##0x7fefffff,#-1)
  r1:0 = r11:10
 }
 {
  r14 = extractu(r28,#2,#22)
  r28 = or(r28,#0x28)
  r5:4 = combine(##0x7ff00000,#0)
 }
 {
  USR = r28
  r14 ^= lsr(r1,#31)
  r28 = r14
 }
 {
  p0 = !cmp.eq(r28,#1)
  p0 = !cmp.eq(r14,#2)
  if (p0.new) r13:12 = r5:4
  p0 = dfcmp.eq(r1:0,r1:0)
 }
 {
  r1:0 = insert(r13:12,#63,#0)
  jumpr r31
 }

.Lmul_abnormal:
 {
  r13:12 = extractu(r1:0,#63,#0)
  r5:4 = extractu(r3:2,#63,#0)
 }
 {
  p3 = cmp.gtu(r13:12,r5:4)
  if (!p3.new) r1:0 = r3:2
  if (!p3.new) r3:2 = r1:0
 }
 {

  p0 = dfclass(r1:0,#0x0f)
  if (!p0.new) jump:nt .Linvalid_nan
  if (!p3) r13:12 = r5:4
  if (!p3) r5:4 = r13:12
 }
 {

  p1 = dfclass(r1:0,#0x08)
  p1 = dfclass(r3:2,#0x0e)
 }
 {


  p0 = dfclass(r1:0,#0x08)
  p0 = dfclass(r3:2,#0x01)
 }
 {
  if (p1) jump .Ltrue_inf
  p2 = dfclass(r3:2,#0x01)
 }
 {
  if (p0) jump .Linvalid_zeroinf
  if (p2) jump .Ltrue_zero
  r28 = ##0x7c000000
 }





 {
  p0 = bitsclr(r1,r28)
  if (p0.new) jump:nt .Lmul_tiny
 }
 {
  r28 = cl0(r5:4)
 }
 {
  r28 = add(r28,#-11)
 }
 {
  r5:4 = asl(r5:4,r28)
 }
 {
  r3:2 = insert(r5:4,#63,#0)
  r1 -= asl(r28,#20)
 }
 jump __hexagon_muldf3
.Lmul_tiny:
 {
  r28 = USR
  r1:0 = xor(r1:0,r3:2)
 }
 {
  r28 = or(r28,#0x30)
  r1:0 = insert(r9:8,#63,#0)
  r5 = extractu(r28,#2,#22)
 }
 {
  USR = r28
  p0 = cmp.gt(r5,#1)
  if (!p0.new) r0 = #0
  r5 ^= lsr(r1,#31)
 }
 {
  p0 = cmp.eq(r5,#3)
  if (!p0.new) r0 = #0
  jumpr r31
 }
.Linvalid_zeroinf:
 {
  r28 = USR
 }
 {
  r1:0 = #-1
  r28 = or(r28,#2)
 }
 {
  USR = r28
 }
 {
  p0 = dfcmp.uo(r1:0,r1:0)
  jumpr r31
 }
.Linvalid_nan:
 {
  p0 = dfclass(r3:2,#0x0f)
  r28 = convert_df2sf(r1:0)
  if (p0.new) r3:2 = r1:0
 }
 {
  r2 = convert_df2sf(r3:2)
  r1:0 = #-1
  jumpr r31
 }
 .falign
.Ltrue_zero:
 {
  r1:0 = r3:2
  r3:2 = r1:0
 }
.Ltrue_inf:
 {
  r3 = extract(r3,#1,#31)
 }
 {
  r1 ^= asl(r3,#31)
  jumpr r31
 }
.size __hexagon_muldf3,.-__hexagon_muldf3
