SECTIONS
{
    .static : ALIGN(4)
    {
        KEEP(*(.rodata.STATIC));
    }
}
