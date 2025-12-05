  .text






  .globl hexagon_memcpy_forward_vp4cp4n2
  .balign 32
  .type hexagon_memcpy_forward_vp4cp4n2,@function
hexagon_memcpy_forward_vp4cp4n2:




  {
    r3 = sub(##4096, r1)
    r5 = lsr(r2, #3)
  }
  {


    r3 = extractu(r3, #10, #2)
    r4 = extractu(r3, #7, #5)
  }
  {
    r3 = minu(r2, r3)
    r4 = minu(r5, r4)
  }
  {
    r4 = or(r4, ##2105344)
    p0 = cmp.eq(r3, #0)
    if (p0.new) jump:nt .Lskipprolog
  }
    l2fetch(r1, r4)
  {
    loop0(.Lprolog, r3)
    r2 = sub(r2, r3)
  }
  .falign
.Lprolog:
  {
    r4 = memw(r1++#4)
    memw(r0++#4) = r4.new
  } :endloop0
.Lskipprolog:
  {

    r3 = lsr(r2, #10)
    if (cmp.eq(r3.new, #0)) jump:nt .Lskipmain
  }
  {
    loop1(.Lout, r3)
    r2 = extractu(r2, #10, #0)
    r3 = ##2105472
  }

  .falign
.Lout:

    l2fetch(r1, r3)
    loop0(.Lpage, #512)
  .falign
.Lpage:
    r5:4 = memd(r1++#8)
  {
    memw(r0++#8) = r4
    memw(r0+#4) = r5
  } :endloop0:endloop1
.Lskipmain:
  {
    r3 = ##2105344
    r4 = lsr(r2, #3)
    p0 = cmp.eq(r2, #0)
    if (p0.new) jumpr:nt r31
  }
  {
    r3 = or(r3, r4)
    loop0(.Lepilog, r2)
  }
    l2fetch(r1, r3)
  .falign
.Lepilog:
  {
    r4 = memw(r1++#4)
    memw(r0++#4) = r4.new
  } :endloop0

    jumpr r31

.size hexagon_memcpy_forward_vp4cp4n2, . - hexagon_memcpy_forward_vp4cp4n2
