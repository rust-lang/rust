 .text
 .global __hexagon_fmadf4
        .type __hexagon_fmadf4,@function
 .global __hexagon_fmadf5
        .type __hexagon_fmadf5,@function
 .global __qdsp_fmadf5 ; .set __qdsp_fmadf5, __hexagon_fmadf5
 .p2align 5
__hexagon_fmadf4:
__hexagon_fmadf5:
fma:
 {
  p0 = dfclass(r1:0,#2)
  p0 = dfclass(r3:2,#2)
  r13:12 = #0
  r15:14 = #0
 }
 {
  r13:12 = insert(r1:0,#52,#11 -3)
  r15:14 = insert(r3:2,#52,#11 -3)
  r7 = ##0x10000000
  allocframe(#32)
 }
 {
  r9:8 = mpyu(r12,r14)
  if (!p0) jump .Lfma_abnormal_ab
  r13 = or(r13,r7)
  r15 = or(r15,r7)
 }
 {
  p0 = dfclass(r5:4,#2)
  if (!p0.new) jump:nt .Lfma_abnormal_c
  r11:10 = combine(r7,#0)
  r7:6 = combine(#0,r9)
 }
.Lfma_abnormal_c_restart:
 {
  r7:6 += mpyu(r14,r13)
  r11:10 = insert(r5:4,#52,#11 -3)
  memd(r29+#0) = r17:16
  memd(r29+#8) = r19:18
 }
 {
  r7:6 += mpyu(r12,r15)
  r19:18 = neg(r11:10)
  p0 = cmp.gt(r5,#-1)
  r28 = xor(r1,r3)
 }
 {
  r18 = extractu(r1,#11,#20)
  r19 = extractu(r3,#11,#20)
  r17:16 = combine(#0,r7)
  if (!p0) r11:10 = r19:18
 }
 {
  r17:16 += mpyu(r13,r15)
  r9:8 = combine(r6,r8)
  r18 = add(r18,r19)




  r19 = extractu(r5,#11,#20)
 }
 {
  r18 = add(r18,#-1023 +(4))
  p3 = !cmp.gt(r28,#-1)
  r7:6 = #0
  r15:14 = #0
 }
 {
  r7:6 = sub(r7:6,r9:8,p3):carry
  p0 = !cmp.gt(r28,#-1)
  p1 = cmp.gt(r19,r18)
  if (p1.new) r19:18 = combine(r18,r19)
 }
 {
  r15:14 = sub(r15:14,r17:16,p3):carry
  if (p0) r9:8 = r7:6




  r7:6 = #0
  r19 = sub(r18,r19)
 }
 {
  if (p0) r17:16 = r15:14
  p0 = cmp.gt(r19,#63)
  if (p1) r9:8 = r7:6
  if (p1) r7:6 = r9:8
 }







 {
  if (p1) r17:16 = r11:10
  if (p1) r11:10 = r17:16
  if (p0) r19 = add(r19,#-64)
  r28 = #63
 }
 {

  if (p0) r7:6 = r11:10
  r28 = asr(r11,#31)
  r13 = min(r19,r28)
  r12 = #0
 }






 {
  if (p0) r11:10 = combine(r28,r28)
  r5:4 = extract(r7:6,r13:12)
  r7:6 = lsr(r7:6,r13)
  r12 = sub(#64,r13)
 }
 {
  r15:14 = #0
  r28 = #-2
  r7:6 |= lsl(r11:10,r12)
  r11:10 = asr(r11:10,r13)
 }
 {
  p3 = cmp.gtu(r5:4,r15:14)
  if (p3.new) r6 = and(r6,r28)



  r15:14 = #1
  r5:4 = #0
 }
 {
  r9:8 = add(r7:6,r9:8,p3):carry
 }
 {
  r17:16 = add(r11:10,r17:16,p3):carry
  r28 = #62
 }







 {
  r12 = add(clb(r17:16),#-2)
  if (!cmp.eq(r12.new,r28)) jump:t 1f
 }

 {
  r11:10 = extractu(r9:8,#62,#2)
  r9:8 = asl(r9:8,#62)
  r18 = add(r18,#-62)
 }
 {
  r17:16 = insert(r11:10,#62,#0)
 }
 {
  r12 = add(clb(r17:16),#-2)
 }
 .falign
1:
 {
  r11:10 = asl(r17:16,r12)
  r5:4 |= asl(r9:8,r12)
  r13 = sub(#64,r12)
  r18 = sub(r18,r12)
 }
 {
  r11:10 |= lsr(r9:8,r13)
  p2 = cmp.gtu(r15:14,r5:4)
  r28 = #1023 +1023 -2
 }
 {
  if (!p2) r10 = or(r10,r14)

  p0 = !cmp.gt(r18,r28)
  p0 = cmp.gt(r18,#1)
  if (!p0.new) jump:nt .Lfma_ovf_unf
 }
 {

  p0 = cmp.gtu(r15:14,r11:10)
  r1:0 = convert_d2df(r11:10)
  r18 = add(r18,#-1023 -60)
  r17:16 = memd(r29+#0)
 }
 {
  r1 += asl(r18,#20)
  r19:18 = memd(r29+#8)
  if (!p0) dealloc_return
 }
.Ladd_yields_zero:

 {
  r28 = USR
  r1:0 = #0
 }
 {
  r28 = extractu(r28,#2,#22)
  r17:16 = memd(r29+#0)
  r19:18 = memd(r29+#8)
 }
 {
  p0 = cmp.eq(r28,#2)
  if (p0.new) r1 = ##0x80000000
  dealloc_return
 }
.Lfma_ovf_unf:
 {
  p0 = cmp.gtu(r15:14,r11:10)
  if (p0.new) jump:nt .Ladd_yields_zero
 }
 {
  r1:0 = convert_d2df(r11:10)
  r18 = add(r18,#-1023 -60)
  r28 = r18
 }


 {
  r1 += asl(r18,#20)
  r7 = extractu(r1,#11,#20)
 }
 {
  r6 = add(r18,r7)
  r17:16 = memd(r29+#0)
  r19:18 = memd(r29+#8)
  r9:8 = abs(r11:10)
 }
 {
  p0 = cmp.gt(r6,##1023 +1023)
  if (p0.new) jump:nt .Lfma_ovf
 }
 {
  p0 = cmp.gt(r6,#0)
  if (p0.new) jump:nt .Lpossible_unf0
 }
 {



  r7 = add(clb(r9:8),#-2)
  r6 = sub(#1+5,r28)
  p3 = cmp.gt(r11,#-1)
 }



 {
  r6 = add(r6,r7)
  r9:8 = asl(r9:8,r7)
  r1 = USR
  r28 = #63
 }
 {
  r7 = min(r6,r28)
  r6 = #0
  r0 = #0x0030
 }
 {
  r3:2 = extractu(r9:8,r7:6)
  r9:8 = asr(r9:8,r7)
 }
 {
  p0 = cmp.gtu(r15:14,r3:2)
  if (!p0.new) r8 = or(r8,r14)
  r9 = setbit(r9,#20 +3)
 }
 {
  r11:10 = neg(r9:8)
  p1 = bitsclr(r8,#(1<<3)-1)
  if (!p1.new) r1 = or(r1,r0)
  r3:2 = #0
 }
 {
  if (p3) r11:10 = r9:8
  USR = r1
  r28 = #-1023 -(52 +3)
 }
 {
  r1:0 = convert_d2df(r11:10)
 }
 {
  r1 += asl(r28,#20)
  dealloc_return
 }
.Lpossible_unf0:
 {
  r28 = ##0x7fefffff
  r9:8 = abs(r11:10)
 }
 {
  p0 = cmp.eq(r0,#0)
  p0 = bitsclr(r1,r28)
  if (!p0.new) dealloc_return:t
  r28 = #0x7fff
 }
 {
  p0 = bitsset(r9,r28)
  r3 = USR
  r2 = #0x0030
 }
 {
  if (p0) r3 = or(r3,r2)
 }
 {
  USR = r3
 }
 {
  p0 = dfcmp.eq(r1:0,r1:0)
  dealloc_return
 }
.Lfma_ovf:
 {
  r28 = USR
  r11:10 = combine(##0x7fefffff,#-1)
  r1:0 = r11:10
 }
 {
  r9:8 = combine(##0x7ff00000,#0)
  r3 = extractu(r28,#2,#22)
  r28 = or(r28,#0x28)
 }
 {
  USR = r28
  r3 ^= lsr(r1,#31)
  r2 = r3
 }
 {
  p0 = !cmp.eq(r2,#1)
  p0 = !cmp.eq(r3,#2)
 }
 {
  p0 = dfcmp.eq(r9:8,r9:8)
  if (p0.new) r11:10 = r9:8
 }
 {
  r1:0 = insert(r11:10,#63,#0)
  dealloc_return
 }
.Lfma_abnormal_ab:
 {
  r9:8 = extractu(r1:0,#63,#0)
  r11:10 = extractu(r3:2,#63,#0)
  deallocframe
 }
 {
  p3 = cmp.gtu(r9:8,r11:10)
  if (!p3.new) r1:0 = r3:2
  if (!p3.new) r3:2 = r1:0
 }
 {
  p0 = dfclass(r1:0,#0x0f)
  if (!p0.new) jump:nt .Lnan
  if (!p3) r9:8 = r11:10
  if (!p3) r11:10 = r9:8
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
  if (p1) jump .Lab_inf
  p2 = dfclass(r3:2,#0x01)
 }
 {
  if (p0) jump .Linvalid
  if (p2) jump .Lab_true_zero
  r28 = ##0x7c000000
 }





 {
  p0 = bitsclr(r1,r28)
  if (p0.new) jump:nt .Lfma_ab_tiny
 }
 {
  r28 = add(clb(r11:10),#-11)
 }
 {
  r11:10 = asl(r11:10,r28)
 }
 {
  r3:2 = insert(r11:10,#63,#0)
  r1 -= asl(r28,#20)
 }
 jump fma

.Lfma_ab_tiny:
 r9:8 = combine(##0x00100000,#0)
 {
  r1:0 = insert(r9:8,#63,#0)
  r3:2 = insert(r9:8,#63,#0)
 }
 jump fma

.Lab_inf:
 {
  r3:2 = lsr(r3:2,#63)
  p0 = dfclass(r5:4,#0x10)
 }
 {
  r1:0 ^= asl(r3:2,#63)
  if (p0) jump .Lnan
 }
 {
  p1 = dfclass(r5:4,#0x08)
  if (p1.new) jump:nt .Lfma_inf_plus_inf
 }

 {
  jumpr r31
 }
 .falign
.Lfma_inf_plus_inf:
 {
  p0 = dfcmp.eq(r1:0,r5:4)
  if (!p0.new) jump:nt .Linvalid
 }
 {
  jumpr r31
 }

.Lnan:
 {
  p0 = dfclass(r3:2,#0x10)
  p1 = dfclass(r5:4,#0x10)
  if (!p0.new) r3:2 = r1:0
  if (!p1.new) r5:4 = r1:0
 }
 {
  r3 = convert_df2sf(r3:2)
  r2 = convert_df2sf(r5:4)
 }
 {
  r3 = convert_df2sf(r1:0)
  r1:0 = #-1
  jumpr r31
 }

.Linvalid:
 {
  r28 = ##0x7f800001
 }
 {
  r1:0 = convert_sf2df(r28)
  jumpr r31
 }

.Lab_true_zero:

 {
  p0 = dfclass(r5:4,#0x10)
  if (p0.new) jump:nt .Lnan
  if (p0.new) r1:0 = r5:4
 }
 {
  p0 = dfcmp.eq(r3:2,r5:4)
  r1 = lsr(r1,#31)
 }
 {
  r3 ^= asl(r1,#31)
  if (!p0) r1:0 = r5:4
  if (!p0) jumpr r31
 }

 {
  p0 = cmp.eq(r3:2,r5:4)
  if (p0.new) jumpr:t r31
  r1:0 = r3:2
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




 .falign
.Lfma_abnormal_c:


 {
  p0 = dfclass(r5:4,#0x10)
  if (p0.new) jump:nt .Lnan
  if (p0.new) r1:0 = r5:4
  deallocframe
 }
 {
  p0 = dfclass(r5:4,#0x08)
  if (p0.new) r1:0 = r5:4
  if (p0.new) jumpr:nt r31
 }


 {
  p0 = dfclass(r5:4,#0x01)
  if (p0.new) jump:nt __hexagon_muldf3
  r28 = #1
 }


 {
  allocframe(#32)
  r11:10 = #0
  r5 = insert(r28,#11,#20)
  jump .Lfma_abnormal_c_restart
 }
.size fma,.-fma
