#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void swap(void *A, void *B, size_t Size) {
  unsigned char Tmp[Size];
  memcpy(Tmp, A, Size);
  memcpy(A, B, Size);
  memcpy(B, Tmp, Size);
}

int partition(void *Base, int Low, int High, size_t Size,
              int (*Cmp)(const void *, const void *)) {
  char *Arr = (char *)Base;

  void *Pivot = Arr + Low * Size;
  int i = Low;
  int j = High;

  while (i < j) {
    while (i <= High - 1 && Cmp(Arr + i * Size, Pivot) <= 0)
      i++;

    while (j >= Low + 1 && Cmp(Arr + j * Size, Pivot) > 0)
      j--;

    if (i < j)
      swap(Arr + i * Size, Arr + j * Size, Size);
  }

  swap(Arr + Low * Size, Arr + j * Size, Size);
  return j;
}

void quickSortRec(void *Base, int Low, int High, size_t Size,
                  int (*Cmp)(const void *, const void *)) {
  if (Low < High) {
    int Part = partition(Base, Low, High, Size, Cmp);
    quickSortRec(Base, Low, Part - 1, Size, Cmp);
    quickSortRec(Base, Part + 1, High, Size, Cmp);
  }
}

void quickSort(void *Base, size_t N, size_t Size,
               int (*Cmp)(const void *, const void *)) {
  if (Size != sizeof(int32_t))
    abort();
  if (N > 1)
    quickSortRec(Base, 0, (int)N - 1, Size, Cmp);
}
