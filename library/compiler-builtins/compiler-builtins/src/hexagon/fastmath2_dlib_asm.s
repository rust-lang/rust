        .text
        .global __hexagon_fast2_dadd_asm
        .type __hexagon_fast2_dadd_asm, @function
__hexagon_fast2_dadd_asm:
        .falign
      {
        R7:6 = VABSDIFFH(R1:0, R3:2)
        R9 = #62
        R4 = SXTH(R0)
        R5 = SXTH(R2)
      } {
        R6 = SXTH(R6)
        P0 = CMP.GT(R4, R5);
        if ( P0.new) R8 = add(R4, #1)
        if (!P0.new) R8 = add(R5, #1)
      } {
        if ( P0) R4 = #1
        if (!P0) R5 = #1
        R0.L = #0
        R6 = MIN(R6, R9)
      } {
        if (!P0) R4 = add(R6, #1)
        if ( P0) R5 = add(R6, #1)
        R2.L = #0
        R11:10 = #0
      } {
        R1:0 = ASR(R1:0, R4)
        R3:2 = ASR(R3:2, R5)
      } {
        R1:0 = add(R1:0, R3:2)
        R10.L = #0x8001
      } {
        R4 = clb(R1:0)
        R9 = #58
      } {
        R4 = add(R4, #-1)
        p0 = cmp.gt(R4, R9)
      } {
        R1:0 = ASL(R1:0, R4)
        R8 = SUB(R8, R4)
        if(p0) jump .Ldenorma
      } {
        R0 = insert(R8, #16, #0)
        jumpr r31
      }
.Ldenorma:
      {
        R1:0 = R11:10
        jumpr r31
      }
        .text
        .global __hexagon_fast2_dsub_asm
        .type __hexagon_fast2_dsub_asm, @function
__hexagon_fast2_dsub_asm:
        .falign
      {
        R7:6 = VABSDIFFH(R1:0, R3:2)
        R9 = #62
        R4 = SXTH(R0)
        R5 = SXTH(R2)
      } {
        R6 = SXTH(R6)
        P0 = CMP.GT(R4, R5);
        if ( P0.new) R8 = add(R4, #1)
        if (!P0.new) R8 = add(R5, #1)
      } {
        if ( P0) R4 = #1
        if (!P0) R5 = #1
        R0.L = #0
        R6 = MIN(R6, R9)
      } {
        if (!P0) R4 = add(R6, #1)
        if ( P0) R5 = add(R6, #1)
        R2.L = #0
        R11:10 = #0
      } {
        R1:0 = ASR(R1:0, R4)
        R3:2 = ASR(R3:2, R5)
      } {
        R1:0 = sub(R1:0, R3:2)
        R10.L = #0x8001
      } {
        R4 = clb(R1:0)
        R9 = #58
      } {
        R4 = add(R4, #-1)
        p0 = cmp.gt(R4, R9)
      } {
        R1:0 = ASL(R1:0, R4)
        R8 = SUB(R8, R4)
        if(p0) jump .Ldenorm
      } {
        R0 = insert(R8, #16, #0)
        jumpr r31
      }
.Ldenorm:
      {
        R1:0 = R11:10
        jumpr r31
      }
        .text
        .global __hexagon_fast2_dmpy_asm
        .type __hexagon_fast2_dmpy_asm, @function
__hexagon_fast2_dmpy_asm:
        .falign
      {
        R13= lsr(R2, #16)
        R5 = sxth(R2)
        R4 = sxth(R0)
        R12= lsr(R0, #16)
      }
      {
        R11:10 = mpy(R1, R3)
        R7:6 = mpy(R1, R13)
        R0.L = #0x0
        R15:14 = #0
      }
      {
        R11:10 = add(R11:10, R11:10)
        R7:6 += mpy(R3, R12)
        R2.L = #0x0
        R15.H = #0x8000
      }
      {
        R7:6 = asr(R7:6, #15)
        R12.L = #0x8001
        p1 = cmp.eq(R1:0, R3:2)
      }
      {
        R7:6 = add(R7:6, R11:10)
        R8 = add(R4, R5)
        p2 = cmp.eq(R1:0, R15:14)
      }
      {
        R9 = clb(R7:6)
        R3:2 = abs(R7:6)
        R11 = #58
      }
      {
        p1 = and(p1, p2)
        R8 = sub(R8, R9)
        R9 = add(R9, #-1)
 p0 = cmp.gt(R9, R11)
      }
      {
        R8 = add(R8, #1)
        R1:0 = asl(R7:6, R9)
        if(p1) jump .Lsat
      }
      {
        R0 = insert(R8,#16, #0)
        if(!p0) jumpr r31
      }
      {
        R0 = insert(R12,#16, #0)
        jumpr r31
      }
.Lsat:
      {
        R1:0 = #-1
      }
      {
        R1:0 = lsr(R1:0, #1)
      }
      {
        R0 = insert(R8,#16, #0)
        jumpr r31
      }
        .text
        .global __hexagon_fast2_qd2f_asm
        .type __hexagon_fast2_qd2f_asm, @function
__hexagon_fast2_qd2f_asm:
      .falign
     {
       R3 = abs(R1):sat
       R4 = sxth(R0)
       R5 = #0x40
       R6.L = #0xffc0
     }
     {
       R0 = extractu(R3, #8, #0)
       p2 = cmp.gt(R4, #126)
       p3 = cmp.ge(R4, #-126)
       R6.H = #0x7fff
     }
     {
       p1 = cmp.eq(R0,#0x40)
       if(p1.new) R5 = #0
       R4 = add(R4, #126)
       if(!p3) jump .Lmin
     }
     {
       p0 = bitsset(R3, R6)
       R0.L = #0x0000
       R2 = add(R3, R5)
       R7 = lsr(R6, #8)
     }
     {
       if(p0) R4 = add(R4, #1)
       if(p0) R3 = #0
       R2 = lsr(R2, #7)
       R0.H = #0x8000
     }
     {
       R0 = and(R0, R1)
       R6 &= asl(R4, #23)
       if(!p0) R3 = and(R2, R7)
       if(p2) jump .Lmax
     }
     {
       R0 += add(R6, R3)
       jumpr r31
     }
.Lmax:
     {
       R0.L = #0xffff;
     }
     {
       R0.H = #0x7f7f;
       jumpr r31
     }
.Lmin:
     {
       R0 = #0x0
       jumpr r31
     }
        .text
        .global __hexagon_fast2_f2qd_asm
        .type __hexagon_fast2_f2qd_asm, @function
__hexagon_fast2_f2qd_asm:







        .falign
  {
       R1 = asl(R0, #7)
       p0 = tstbit(R0, #31)
       R5:4 = #0
       R3 = add(R0,R0)
  }
  {
       R1 = setbit(R1, #30)
       R0= extractu(R0,#8,#23)
       R4.L = #0x8001
       p1 = cmp.eq(R3, #0)
  }
  {
       R1= extractu(R1, #31, #0)
       R0= add(R0, #-126)
       R2 = #0
       if(p1) jump .Lminqd
  }
  {
       R0 = zxth(R0)
       if(p0) R1= sub(R2, R1)
       jumpr r31
  }
.Lminqd:
  {
       R1:0 = R5:4
       jumpr r31
  }
