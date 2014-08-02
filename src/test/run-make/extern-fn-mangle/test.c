#include <stdint.h>

uint32_t foo();
uint32_t bar();

uint32_t add() {
	return foo() + bar();
}
