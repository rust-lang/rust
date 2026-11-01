        .text
        .global __hexagon_fast2ldadd_asm
        .type __hexagon_fast2ldadd_asm, @function
__hexagon_fast2ldadd_asm:
        .falign
      {
        R4 = memw(r29+#8)
        R5 = memw(r29+#24)
        r7 = r0
      }
      {
        R6 = sub(R4, R5):sat
        P0 = CMP.GT(R4, R5);
        if ( P0.new) R8 = add(R4, #1)
        if (!P0.new) R8 = add(R5, #1)
      } {
        R6 = abs(R6):sat
        if ( P0) R4 = #1
        if (!P0) R5 = #1
        R9 = #62
      } {
        R6 = MIN(R6, R9)
        R1:0 = memd(r29+#0)
        R3:2 = memd(r29+#16)
      } {
        if (!P0) R4 = add(R6, #1)
        if ( P0) R5 = add(R6, #1)
      } {
        R1:0 = ASR(R1:0, R4)
        R3:2 = ASR(R3:2, R5)
      } {
        R1:0 = add(R1:0, R3:2)
        R3:2 = #0
      } {
        R4 = clb(R1:0)
        R9.L =#0x0001
      } {
        R8 -= add(R4, #-1)
        R4 = add(R4, #-1)
        p0 = cmp.gt(R4, #58)
        R9.H =#0x8000
      } {
        if(!p0)memw(r7+#8) = R8
        R1:0 = ASL(R1:0, R4)
        if(p0) jump .Ldenorma1
      } {
        memd(r7+#0) = R1:0
        jumpr r31
      }
.Ldenorma1:
        memd(r7+#0) = R3:2
      {
        memw(r7+#8) = R9
        jumpr r31
      }
        .text
        .global __hexagon_fast2ldsub_asm
        .type __hexagon_fast2ldsub_asm, @function
__hexagon_fast2ldsub_asm:
        .falign
      {
        R4 = memw(r29+#8)
        R5 = memw(r29+#24)
        r7 = r0
      }
      {
        R6 = sub(R4, R5):sat
        P0 = CMP.GT(R4, R5);
        if ( P0.new) R8 = add(R4, #1)
        if (!P0.new) R8 = add(R5, #1)
      } {
        R6 = abs(R6):sat
        if ( P0) R4 = #1
        if (!P0) R5 = #1
        R9 = #62
      } {
        R6 = min(R6, R9)
        R1:0 = memd(r29+#0)
        R3:2 = memd(r29+#16)
      } {
        if (!P0) R4 = add(R6, #1)
        if ( P0) R5 = add(R6, #1)
      } {
        R1:0 = ASR(R1:0, R4)
        R3:2 = ASR(R3:2, R5)
      } {
        R1:0 = sub(R1:0, R3:2)
        R3:2 = #0
      } {
        R4 = clb(R1:0)
        R9.L =#0x0001
      } {
        R8 -= add(R4, #-1)
        R4 = add(R4, #-1)
        p0 = cmp.gt(R4, #58)
        R9.H =#0x8000
      } {
        if(!p0)memw(r7+#8) = R8
        R1:0 = asl(R1:0, R4)
        if(p0) jump .Ldenorma_s
      } {
        memd(r7+#0) = R1:0
        jumpr r31
      }
.Ldenorma_s:
        memd(r7+#0) = R3:2
      {
        memw(r7+#8) = R9
        jumpr r31
      }
        .text
        .global __hexagon_fast2ldmpy_asm
        .type __hexagon_fast2ldmpy_asm, @function
__hexagon_fast2ldmpy_asm:
        .falign
      {
        R15:14 = memd(r29+#0)
        R3:2 = memd(r29+#16)
        R13:12 = #0
      }
      {
        R8= extractu(R2, #31, #1)
        R9= extractu(R14, #31, #1)
        R13.H = #0x8000
      }
      {
        R11:10 = mpy(R15, R3)
        R7:6 = mpy(R15, R8)
        R4 = memw(r29+#8)
        R5 = memw(r29+#24)
      }
      {
        R11:10 = add(R11:10, R11:10)
        R7:6 += mpy(R3, R9)
      }
      {
        R7:6 = asr(R7:6, #30)
        R8.L = #0x0001
        p1 = cmp.eq(R15:14, R3:2)
      }
      {
        R7:6 = add(R7:6, R11:10)
        R4= add(R4, R5)
        p2 = cmp.eq(R3:2, R13:12)
      }
      {
        R9 = clb(R7:6)
        R8.H = #0x8000
        p1 = and(p1, p2)
      }
      {
        R4-= add(R9, #-1)
        R9 = add(R9, #-1)
        if(p1) jump .Lsat1
      }
      {
        R7:6 = asl(R7:6, R9)
        memw(R0+#8) = R4
 p0 = cmp.gt(R9, #58)
        if(p0.new) jump:NT .Ldenorm1
      }
      {
        memd(R0+#0) = R7:6
        jumpr r31
      }
.Lsat1:
      {
        R13:12 = #0
        R4+= add(R9, #1)
      }
      {
        R13.H = #0x4000
        memw(R0+#8) = R4
      }
      {
        memd(R0+#0) = R13:12
        jumpr r31
      }
.Ldenorm1:
      {
        memw(R0+#8) = R8
        R15:14 = #0
      }
      {
        memd(R0+#0) = R15:14
        jumpr r31
      }
