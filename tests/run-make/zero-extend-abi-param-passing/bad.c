#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


struct bloc {
    uint16_t a;
    uint16_t b;
    uint16_t c;
};

uint16_t c_read_value(uint32_t a, uint32_t b, uint32_t c) {
    struct bloc *data = malloc(sizeof(struct bloc));

    data->a = a & 0xFFFF;
    data->b = b & 0xFFFF;
    data->c = c & 0xFFFF;

    printf("C struct: a = %u, b = %u, c = %u\n",
        (unsigned) data->a, (unsigned) data->b, (unsigned) data->c);
    printf("C function returns %u\n", (unsigned) data->b);

    return data->b; /* leak data */
}
