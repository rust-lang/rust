 .text
 .global __hexagon_adddf3
 .global __hexagon_subdf3
 .type __hexagon_adddf3, @function
 .type __hexagon_subdf3, @function

.global __qdsp_adddf3 ; .set __qdsp_adddf3, __hexagon_adddf3
.global __hexagon_fast_adddf3 ; .set __hexagon_fast_adddf3, __hexagon_adddf3
.global __hexagon_fast2_adddf3 ; .set __hexagon_fast2_adddf3, __hexagon_adddf3
.global __qdsp_subdf3 ; .set __qdsp_subdf3, __hexagon_subdf3
.global __hexagon_fast_subdf3 ; .set __hexagon_fast_subdf3, __hexagon_subdf3
.global __hexagon_fast2_subdf3 ; .set __hexagon_fast2_subdf3, __hexagon_subdf3

 .p2align 5
__hexagon_adddf3:
 {
  r4 = extractu(r1,#11,#20)
  r5 = extractu(r3,#11,#20)
  r13:12 = combine(##0x20000000,#0)
 }
 {
  p3 = dfclass(r1:0,#2)
  p3 = dfclass(r3:2,#2)
  r9:8 = r13:12
  p2 = cmp.gtu(r5,r4)
 }
 {
  if (!p3) jump .Ladd_abnormal
  if (p2) r1:0 = r3:2
  if (p2) r3:2 = r1:0
  if (p2) r5:4 = combine(r4,r5)
 }
 {
  r13:12 = insert(r1:0,#52,#11 -2)
  r9:8 = insert(r3:2,#52,#11 -2)
  r15 = sub(r4,r5)
  r7:6 = combine(#62,#1)
 }





.Ladd_continue:
 {
  r15 = min(r15,r7)

  r11:10 = neg(r13:12)
  p2 = cmp.gt(r1,#-1)
  r14 = #0
 }
 {
  if (!p2) r13:12 = r11:10
  r11:10 = extractu(r9:8,r15:14)
  r9:8 = ASR(r9:8,r15)




  r15:14 = #0
 }
 {
  p1 = cmp.eq(r11:10,r15:14)
  if (!p1.new) r8 = or(r8,r6)
  r5 = add(r4,#-1024 -60)
  p3 = cmp.gt(r3,#-1)
 }
 {
  r13:12 = add(r13:12,r9:8)
  r11:10 = sub(r13:12,r9:8)
  r7:6 = combine(#54,##2045)
 }
 {
  p0 = cmp.gtu(r4,r7)
  p0 = !cmp.gtu(r4,r6)
  if (!p0.new) jump:nt .Ladd_ovf_unf
  if (!p3) r13:12 = r11:10
 }
 {
  r1:0 = convert_d2df(r13:12)
  p0 = cmp.eq(r13,#0)
  p0 = cmp.eq(r12,#0)
  if (p0.new) jump:nt .Ladd_zero
 }
 {
  r1 += asl(r5,#20)
  jumpr r31
 }
 .falign
__hexagon_subdf3:
 {
  r3 = togglebit(r3,#31)
  jump __qdsp_adddf3
 }


 .falign
.Ladd_zero:


 {
  r28 = USR
  r1:0 = #0
  r3 = #1
 }
 {
  r28 = extractu(r28,#2,#22)
  r3 = asl(r3,#31)
 }
 {
  p0 = cmp.eq(r28,#2)
  if (p0.new) r1 = xor(r1,r3)
  jumpr r31
 }
 .falign
.Ladd_ovf_unf:
 {
  r1:0 = convert_d2df(r13:12)
  p0 = cmp.eq(r13,#0)
  p0 = cmp.eq(r12,#0)
  if (p0.new) jump:nt .Ladd_zero
 }
 {
  r28 = extractu(r1,#11,#20)
  r1 += asl(r5,#20)
 }
 {
  r5 = add(r5,r28)
  r3:2 = combine(##0x00100000,#0)
 }
 {
  p0 = cmp.gt(r5,##1024 +1024 -2)
  if (p0.new) jump:nt .Ladd_ovf
 }
 {
  p0 = cmp.gt(r5,#0)
  if (p0.new) jumpr:t r31
  r28 = sub(#1,r5)
 }
 {
  r3:2 = insert(r1:0,#52,#0)
  r1:0 = r13:12
 }
 {
  r3:2 = lsr(r3:2,r28)
 }
 {
  r1:0 = insert(r3:2,#63,#0)
  jumpr r31
 }
 .falign
.Ladd_ovf:

 {
  r1:0 = r13:12
  r28 = USR
  r13:12 = combine(##0x7fefffff,#-1)
 }
 {
  r5 = extractu(r28,#2,#22)
  r28 = or(r28,#0x28)
  r9:8 = combine(##0x7ff00000,#0)
 }
 {
  USR = r28
  r5 ^= lsr(r1,#31)
  r28 = r5
 }
 {
  p0 = !cmp.eq(r28,#1)
  p0 = !cmp.eq(r5,#2)
  if (p0.new) r13:12 = r9:8
 }
 {
  r1:0 = insert(r13:12,#63,#0)
 }
 {
  p0 = dfcmp.eq(r1:0,r1:0)
  jumpr r31
 }

.Ladd_abnormal:
 {
  r13:12 = extractu(r1:0,#63,#0)
  r9:8 = extractu(r3:2,#63,#0)
 }
 {
  p3 = cmp.gtu(r13:12,r9:8)
  if (!p3.new) r1:0 = r3:2
  if (!p3.new) r3:2 = r1:0
 }
 {

  p0 = dfclass(r1:0,#0x0f)
  if (!p0.new) jump:nt .Linvalid_nan_add
  if (!p3) r13:12 = r9:8
  if (!p3) r9:8 = r13:12
 }
 {


  p1 = dfclass(r1:0,#0x08)
  if (p1.new) jump:nt .Linf_add
 }
 {
  p2 = dfclass(r3:2,#0x01)
  if (p2.new) jump:nt .LB_zero
  r13:12 = #0
 }

 {
  p0 = dfclass(r1:0,#4)
  if (p0.new) jump:nt .Ladd_two_subnormal
  r13:12 = combine(##0x20000000,#0)
 }
 {
  r4 = extractu(r1,#11,#20)
  r5 = #1

  r9:8 = asl(r9:8,#11 -2)
 }



 {
  r13:12 = insert(r1:0,#52,#11 -2)
  r15 = sub(r4,r5)
  r7:6 = combine(#62,#1)
  jump .Ladd_continue
 }

.Ladd_two_subnormal:
 {
  r13:12 = extractu(r1:0,#63,#0)
  r9:8 = extractu(r3:2,#63,#0)
 }
 {
  r13:12 = neg(r13:12)
  r9:8 = neg(r9:8)
  p0 = cmp.gt(r1,#-1)
  p1 = cmp.gt(r3,#-1)
 }
 {
  if (p0) r13:12 = r1:0
  if (p1) r9:8 = r3:2
 }
 {
  r13:12 = add(r13:12,r9:8)
 }
 {
  r9:8 = neg(r13:12)
  p0 = cmp.gt(r13,#-1)
  r3:2 = #0
 }
 {
  if (!p0) r1:0 = r9:8
  if (p0) r1:0 = r13:12
  r3 = ##0x80000000
 }
 {
  if (!p0) r1 = or(r1,r3)
  p0 = dfcmp.eq(r1:0,r3:2)
  if (p0.new) jump:nt .Lzero_plus_zero
 }
 {
  jumpr r31
 }

.Linvalid_nan_add:
 {
  r28 = convert_df2sf(r1:0)
  p0 = dfclass(r3:2,#0x0f)
  if (p0.new) r3:2 = r1:0
 }
 {
  r2 = convert_df2sf(r3:2)
  r1:0 = #-1
  jumpr r31
 }
 .falign
.LB_zero:
 {
  p0 = dfcmp.eq(r13:12,r1:0)
  if (!p0.new) jumpr:t r31
 }




.Lzero_plus_zero:
 {
  p0 = cmp.eq(r1:0,r3:2)
  if (p0.new) jumpr:t r31
 }
 {
  r28 = USR
 }
 {
  r28 = extractu(r28,#2,#22)
  r1:0 = #0
 }
 {
  p0 = cmp.eq(r28,#2)
  if (p0.new) r1 = ##0x80000000
  jumpr r31
 }
.Linf_add:

 {
  p0 = !cmp.eq(r1,r3)
  p0 = dfclass(r3:2,#8)
  if (!p0.new) jumpr:t r31
 }
 {
  r2 = ##0x7f800001
 }
 {
  r1:0 = convert_sf2df(r2)
  jumpr r31
 }
.size __hexagon_adddf3,.-__hexagon_adddf3
