 .text
 .global __hexagon_divdf3
 .type __hexagon_divdf3,@function
 .global __qdsp_divdf3 ; .set __qdsp_divdf3, __hexagon_divdf3
        .global __hexagon_fast_divdf3 ; .set __hexagon_fast_divdf3, __hexagon_divdf3
        .global __hexagon_fast2_divdf3 ; .set __hexagon_fast2_divdf3, __hexagon_divdf3
 .p2align 5
__hexagon_divdf3:
 {
  p2 = dfclass(r1:0,#0x02)
  p2 = dfclass(r3:2,#0x02)
  r13:12 = combine(r3,r1)
  r28 = xor(r1,r3)
 }
 {
  if (!p2) jump .Ldiv_abnormal
  r7:6 = extractu(r3:2,#23,#52 -23)
  r8 = ##0x3f800001
 }
 {
  r9 = or(r8,r6)
  r13 = extractu(r13,#11,#52 -32)
  r12 = extractu(r12,#11,#52 -32)
  p3 = cmp.gt(r28,#-1)
 }


.Ldenorm_continue:
 {
  r11,p0 = sfrecipa(r8,r9)
  r10 = and(r8,#-2)
  r28 = #1
  r12 = sub(r12,r13)
 }


 {
  r10 -= sfmpy(r11,r9):lib
  r1 = insert(r28,#11 +1,#52 -32)
  r13 = ##0x00800000 << 3
 }
 {
  r11 += sfmpy(r11,r10):lib
  r3 = insert(r28,#11 +1,#52 -32)
  r10 = and(r8,#-2)
 }
 {
  r10 -= sfmpy(r11,r9):lib
  r5 = #-0x3ff +1
  r4 = #0x3ff -1
 }
 {
  r11 += sfmpy(r11,r10):lib
  p1 = cmp.gt(r12,r5)
  p1 = !cmp.gt(r12,r4)
 }
 {
  r13 = insert(r11,#23,#3)
  r5:4 = #0
  r12 = add(r12,#-61)
 }




 {
  r13 = add(r13,#((-3) << 3))
 }
 { r7:6 = mpyu(r13,r1); r1:0 = asl(r1:0,# ( 15 )); }; { r6 = # 0; r1:0 -= mpyu(r7,r2); r15:14 = mpyu(r7,r3); }; { r5:4 += ASL(r7:6, # ( 14 )); r1:0 -= asl(r15:14, # 32); }
 { r7:6 = mpyu(r13,r1); r1:0 = asl(r1:0,# ( 15 )); }; { r6 = # 0; r1:0 -= mpyu(r7,r2); r15:14 = mpyu(r7,r3); }; { r5:4 += ASR(r7:6, # ( 1 )); r1:0 -= asl(r15:14, # 32); }
 { r7:6 = mpyu(r13,r1); r1:0 = asl(r1:0,# ( 15 )); }; { r6 = # 0; r1:0 -= mpyu(r7,r2); r15:14 = mpyu(r7,r3); }; { r5:4 += ASR(r7:6, # ( 16 )); r1:0 -= asl(r15:14, # 32); }
 { r7:6 = mpyu(r13,r1); r1:0 = asl(r1:0,# ( 15 )); }; { r6 = # 0; r1:0 -= mpyu(r7,r2); r15:14 = mpyu(r7,r3); }; { r5:4 += ASR(r7:6, # ( 31 )); r1:0 -= asl(r15:14, # 32); r7:6=# ( 0 ); }







 {

  r15:14 = sub(r1:0,r3:2)
  p0 = cmp.gtu(r3:2,r1:0)

  if (!p0.new) r6 = #2
 }
 {
  r5:4 = add(r5:4,r7:6)
  if (!p0) r1:0 = r15:14
  r15:14 = #0
 }
 {
  p0 = cmp.eq(r1:0,r15:14)
  if (!p0.new) r4 = or(r4,r28)
 }
 {
  r7:6 = neg(r5:4)
 }
 {
  if (!p3) r5:4 = r7:6
 }
 {
  r1:0 = convert_d2df(r5:4)
  if (!p1) jump .Ldiv_ovf_unf
 }
 {
  r1 += asl(r12,#52 -32)
  jumpr r31
 }

.Ldiv_ovf_unf:
 {
  r1 += asl(r12,#52 -32)
  r13 = extractu(r1,#11,#52 -32)
 }
 {
  r7:6 = abs(r5:4)
  r12 = add(r12,r13)
 }
 {
  p0 = cmp.gt(r12,##0x3ff +0x3ff)
  if (p0.new) jump:nt .Ldiv_ovf
 }
 {
  p0 = cmp.gt(r12,#0)
  if (p0.new) jump:nt .Lpossible_unf2
 }
 {
  r13 = add(clb(r7:6),#-1)
  r12 = sub(#7,r12)
  r10 = USR
  r11 = #63
 }
 {
  r13 = min(r12,r11)
  r11 = or(r10,#0x030)
  r7:6 = asl(r7:6,r13)
  r12 = #0
 }
 {
  r15:14 = extractu(r7:6,r13:12)
  r7:6 = lsr(r7:6,r13)
  r3:2 = #1
 }
 {
  p0 = cmp.gtu(r3:2,r15:14)
  if (!p0.new) r6 = or(r2,r6)
  r7 = setbit(r7,#52 -32+4)
 }
 {
  r5:4 = neg(r7:6)
  p0 = bitsclr(r6,#(1<<4)-1)
  if (!p0.new) r10 = r11
 }
 {
  USR = r10
  if (p3) r5:4 = r7:6
  r10 = #-0x3ff -(52 +4)
 }
 {
  r1:0 = convert_d2df(r5:4)
 }
 {
  r1 += asl(r10,#52 -32)
  jumpr r31
 }


.Lpossible_unf2:


 {
  r3:2 = extractu(r1:0,#63,#0)
  r15:14 = combine(##0x00100000,#0)
  r10 = #0x7FFF
 }
 {
  p0 = dfcmp.eq(r15:14,r3:2)
  p0 = bitsset(r7,r10)
 }






 {
  if (!p0) jumpr r31
  r10 = USR
 }

 {
  r10 = or(r10,#0x30)
 }
 {
  USR = r10
 }
 {
  p0 = dfcmp.eq(r1:0,r1:0)
  jumpr r31
 }

.Ldiv_ovf:



 {
  r10 = USR
  r3:2 = combine(##0x7fefffff,#-1)
  r1 = mux(p3,#0,#-1)
 }
 {
  r7:6 = combine(##0x7ff00000,#0)
  r5 = extractu(r10,#2,#22)
  r10 = or(r10,#0x28)
 }
 {
  USR = r10
  r5 ^= lsr(r1,#31)
  r4 = r5
 }
 {
  p0 = !cmp.eq(r4,#1)
  p0 = !cmp.eq(r5,#2)
  if (p0.new) r3:2 = r7:6
  p0 = dfcmp.eq(r3:2,r3:2)
 }
 {
  r1:0 = insert(r3:2,#63,#0)
  jumpr r31
 }







.Ldiv_abnormal:
 {
  p0 = dfclass(r1:0,#0x0F)
  p0 = dfclass(r3:2,#0x0F)
  p3 = cmp.gt(r28,#-1)
 }
 {
  p1 = dfclass(r1:0,#0x08)
  p1 = dfclass(r3:2,#0x08)
 }
 {
  p2 = dfclass(r1:0,#0x01)
  p2 = dfclass(r3:2,#0x01)
 }
 {
  if (!p0) jump .Ldiv_nan
  if (p1) jump .Ldiv_invalid
 }
 {
  if (p2) jump .Ldiv_invalid
 }
 {
  p2 = dfclass(r1:0,#(0x0F ^ 0x01))
  p2 = dfclass(r3:2,#(0x0F ^ 0x08))
 }
 {
  p1 = dfclass(r1:0,#(0x0F ^ 0x08))
  p1 = dfclass(r3:2,#(0x0F ^ 0x01))
 }
 {
  if (!p2) jump .Ldiv_zero_result
  if (!p1) jump .Ldiv_inf_result
 }





 {
  p0 = dfclass(r1:0,#0x02)
  p1 = dfclass(r3:2,#0x02)
  r10 = ##0x00100000
 }
 {
  r13:12 = combine(r3,r1)
  r1 = insert(r10,#11 +1,#52 -32)
  r3 = insert(r10,#11 +1,#52 -32)
 }
 {
  if (p0) r1 = or(r1,r10)
  if (p1) r3 = or(r3,r10)
 }
 {
  r5 = add(clb(r1:0),#-11)
  r4 = add(clb(r3:2),#-11)
  r10 = #1
 }
 {
  r12 = extractu(r12,#11,#52 -32)
  r13 = extractu(r13,#11,#52 -32)
 }
 {
  r1:0 = asl(r1:0,r5)
  r3:2 = asl(r3:2,r4)
  if (!p0) r12 = sub(r10,r5)
  if (!p1) r13 = sub(r10,r4)
 }
 {
  r7:6 = extractu(r3:2,#23,#52 -23)
 }
 {
  r9 = or(r8,r6)
  jump .Ldenorm_continue
 }

.Ldiv_zero_result:
 {
  r1 = xor(r1,r3)
  r3:2 = #0
 }
 {
  r1:0 = insert(r3:2,#63,#0)
  jumpr r31
 }
.Ldiv_inf_result:
 {
  p2 = dfclass(r3:2,#0x01)
  p2 = dfclass(r1:0,#(0x0F ^ 0x08))
 }
 {
  r10 = USR
  if (!p2) jump 1f
  r1 = xor(r1,r3)
 }
 {
  r10 = or(r10,#0x04)
 }
 {
  USR = r10
 }
1:
 {
  r3:2 = combine(##0x7ff00000,#0)
  p0 = dfcmp.uo(r3:2,r3:2)
 }
 {
  r1:0 = insert(r3:2,#63,#0)
  jumpr r31
 }
.Ldiv_nan:
 {
  p0 = dfclass(r1:0,#0x10)
  p1 = dfclass(r3:2,#0x10)
  if (!p0.new) r1:0 = r3:2
  if (!p1.new) r3:2 = r1:0
 }
 {
  r5 = convert_df2sf(r1:0)
  r4 = convert_df2sf(r3:2)
 }
 {
  r1:0 = #-1
  jumpr r31
 }

.Ldiv_invalid:
 {
  r10 = ##0x7f800001
 }
 {
  r1:0 = convert_sf2df(r10)
  jumpr r31
 }
.size __hexagon_divdf3,.-__hexagon_divdf3
