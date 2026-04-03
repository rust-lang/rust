#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_ELEMS 5

void quickSort(void *Base, size_t N, size_t Size,
               int (*Cmp)(const void *, const void *));

#ifdef __cplusplus
}
#endif

int cmpI32Ascending(const void *LHS, const void *RHS) {
  int32_t x = *(const int32_t *)LHS;
  int32_t y = *(const int32_t *)RHS;

  if (x < y)
    return -1;
  else if (x > y)
    return 1;
  else
    return 0;
}

int main() {
  int32_t Data[NUM_ELEMS] = {4, 2, 5, 3, 1};

  printf("Before sorting: ");
  for (int i = 0; i < NUM_ELEMS; i++)
    printf("%d ", Data[i]);
  printf("\n");

  quickSort(Data, NUM_ELEMS, sizeof(int32_t), cmpI32Ascending);

  printf("After sorting:  ");
  for (int i = 0; i < NUM_ELEMS; i++)
    printf("%d ", Data[i]);
  printf("\n");

  for (size_t i = 1; i < NUM_ELEMS; i++)
    if (Data[i - 1] > Data[i])
      return 42;

  return 0;
}

#ifdef __cplusplus
}
#endif
