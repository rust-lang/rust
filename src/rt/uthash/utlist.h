/*
Copyright (c) 2007-2009, Troy D. Hanson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef UTLIST_H
#define UTLIST_H

#define UTLIST_VERSION 1.0

/* C++ requires extra stringent casting */
#if defined __cplusplus
#define LTYPEOF(x) (typeof(x))
#else
#define LTYPEOF(x)
#endif
/* 
 * This file contains macros to manipulate singly and doubly-linked lists.
 *
 * 1. LL_ macros:  singly-linked lists.
 * 2. DL_ macros:  doubly-linked lists.
 * 3. CDL_ macros: circular doubly-linked lists.
 *
 * To use singly-linked lists, your structure must have a "next" pointer.
 * To use doubly-linked lists, your structure must "prev" and "next" pointers.
 * Either way, the pointer to the head of the list must be initialized to NULL.
 * 
 * ----------------.EXAMPLE -------------------------
 * struct item {
 *      int id;
 *      struct item *prev, *next;
 * }
 *
 * struct item *list = NULL:
 *
 * int main() {
 *      struct item *item;
 *      ... allocate and populate item ...
 *      DL_APPEND(list, item);
 * }
 * --------------------------------------------------
 *
 * For doubly-linked lists, the append and delete macros are O(1)
 * For singly-linked lists, append and delete are O(n) but prepend is O(1)
 * The sort macro is O(n log(n)) for all types of single/double/circular lists.
 */

/******************************************************************************
 * The SORT macros                                                            *
 *****************************************************************************/
#define LL_SORT(l,cmp)                                                           \
 LISTSORT(l,0,0,FIELD_OFFSET(l,next),cmp)
#define DL_SORT(l,cmp)                                                           \
 LISTSORT(l,0,FIELD_OFFSET(l,prev),FIELD_OFFSET(l,next),cmp)
#define CDL_SORT(l,cmp)                                                          \
 LISTSORT(l,1,FIELD_OFFSET(l,prev),FIELD_OFFSET(l,next),cmp)

/* The macros can't assume or cast to the caller's list element type. So we use
 * a couple tricks when we need to deal with those element's prev/next pointers.
 * Basically we use char pointer arithmetic to get those field offsets. */
#define FIELD_OFFSET(ptr,field) ((char*)&((ptr)->field) - (char*)(ptr))
#define LNEXT(e,no) (*(char**)(((char*)e) + no))
#define LPREV(e,po) (*(char**)(((char*)e) + po))
/******************************************************************************
 * The LISTSORT macro is an adaptation of Simon Tatham's O(n log(n)) mergesort*
 * Unwieldy variable names used here to avoid shadowing passed-in variables.  *
 *****************************************************************************/
#define LISTSORT(list, is_circular, po, no, cmp)                                 \
do {                                                                             \
  void *_ls_p, *_ls_q, *_ls_e, *_ls_tail, *_ls_oldhead;                          \
  int _ls_insize, _ls_nmerges, _ls_psize, _ls_qsize, _ls_i, _ls_looping;         \
  int _ls_is_double = (po==0) ? 0 : 1;                                           \
  if (list) {                                                                    \
    _ls_insize = 1;                                                              \
    _ls_looping = 1;                                                             \
    while (_ls_looping) {                                                        \
      _ls_p = list;                                                              \
      _ls_oldhead = list;                                                        \
      list = NULL;                                                               \
      _ls_tail = NULL;                                                           \
      _ls_nmerges = 0;                                                           \
      while (_ls_p) {                                                            \
        _ls_nmerges++;                                                           \
        _ls_q = _ls_p;                                                           \
        _ls_psize = 0;                                                           \
        for (_ls_i = 0; _ls_i < _ls_insize; _ls_i++) {                           \
          _ls_psize++;                                                           \
          if (is_circular)  {                                                    \
            _ls_q = ((LNEXT(_ls_q,no) == _ls_oldhead) ? NULL : LNEXT(_ls_q,no)); \
          } else  {                                                              \
            _ls_q = LNEXT(_ls_q,no);                                             \
          }                                                                      \
          if (!_ls_q) break;                                                     \
        }                                                                        \
        _ls_qsize = _ls_insize;                                                  \
        while (_ls_psize > 0 || (_ls_qsize > 0 && _ls_q)) {                      \
          if (_ls_psize == 0) {                                                  \
            _ls_e = _ls_q; _ls_q = LNEXT(_ls_q,no); _ls_qsize--;                 \
            if (is_circular && _ls_q == _ls_oldhead) { _ls_q = NULL; }           \
          } else if (_ls_qsize == 0 || !_ls_q) {                                 \
            _ls_e = _ls_p; _ls_p = LNEXT(_ls_p,no); _ls_psize--;                 \
            if (is_circular && (_ls_p == _ls_oldhead)) { _ls_p = NULL; }         \
          } else if (cmp(LTYPEOF(list)_ls_p,LTYPEOF(list)_ls_q) <= 0) {          \
            _ls_e = _ls_p; _ls_p = LNEXT(_ls_p,no); _ls_psize--;                 \
            if (is_circular && (_ls_p == _ls_oldhead)) { _ls_p = NULL; }         \
          } else {                                                               \
            _ls_e = _ls_q; _ls_q = LNEXT(_ls_q,no); _ls_qsize--;                 \
            if (is_circular && (_ls_q == _ls_oldhead)) { _ls_q = NULL; }         \
          }                                                                      \
          if (_ls_tail) {                                                        \
            LNEXT(_ls_tail,no) = (char*)_ls_e;                                   \
          } else {                                                               \
            list = LTYPEOF(list)_ls_e;                                           \
          }                                                                      \
          if (_ls_is_double) {                                                   \
            LPREV(_ls_e,po) = (char*)_ls_tail;                                   \
          }                                                                      \
          _ls_tail = _ls_e;                                                      \
        }                                                                        \
        _ls_p = _ls_q;                                                           \
      }                                                                          \
      if (is_circular) {                                                         \
        LNEXT(_ls_tail,no) = (char*)list;                                        \
        if (_ls_is_double) {                                                     \
          LPREV(list,po) = (char*)_ls_tail;                                      \
        }                                                                        \
      } else  {                                                                  \
        LNEXT(_ls_tail,no) = NULL;                                               \
      }                                                                          \
      if (_ls_nmerges <= 1) {                                                    \
        _ls_looping=0;                                                           \
      }                                                                          \
      _ls_insize *= 2;                                                           \
    }                                                                            \
  }                                                                              \
} while (0)

/******************************************************************************
 * singly linked list macros (non-circular)                                   *
 *****************************************************************************/
#define LL_PREPEND(head,add)                                                     \
do {                                                                             \
  (add)->next = head;                                                            \
  head = add;                                                                    \
} while (0)

#define LL_APPEND(head,add)                                                      \
do {                                                                             \
  (add)->next=NULL;                                                              \
  if (head) {                                                                    \
    char *_lla_el = (char*)(head);                                               \
    unsigned _lla_no = FIELD_OFFSET(head,next);                                  \
    while (LNEXT(_lla_el,_lla_no)) { _lla_el = LNEXT(_lla_el,_lla_no); }         \
    LNEXT(_lla_el,_lla_no)=(char*)(add);                                         \
  } else {                                                                       \
    (head)=(add);                                                                \
  }                                                                              \
} while (0)

#define LL_DELETE(head,del)                                                      \
do {                                                                             \
  if ((head) == (del)) {                                                         \
    (head)=(head)->next;                                                         \
  } else {                                                                       \
    char *_lld_el = (char*)(head);                                               \
    unsigned _lld_no = FIELD_OFFSET(head,next);                                  \
    while (LNEXT(_lld_el,_lld_no) && (LNEXT(_lld_el,_lld_no) != (char*)(del))) { \
      _lld_el = LNEXT(_lld_el,_lld_no);                                          \
    }                                                                            \
    if (LNEXT(_lld_el,_lld_no)) {                                                \
      LNEXT(_lld_el,_lld_no) = (char*)((del)->next);                             \
    }                                                                            \
  }                                                                              \
} while (0)

#define LL_FOREACH(head,el)                                                      \
    for(el=head;el;el=el->next)

/******************************************************************************
 * doubly linked list macros (non-circular)                                   *
 *****************************************************************************/
#define DL_PREPEND(head,add)                                                     \
do {                                                                             \
 (add)->next = head;                                                             \
 if (head) {                                                                     \
   (add)->prev = (head)->prev;                                                   \
   (head)->prev = (add);                                                         \
 } else {                                                                        \
   (add)->prev = (add);                                                          \
 }                                                                               \
 (head) = (add);                                                                 \
} while (0)

#define DL_APPEND(head,add)                                                      \
do {                                                                             \
  if (head) {                                                                    \
      (add)->prev = (head)->prev;                                                \
      (head)->prev->next = (add);                                                \
      (head)->prev = (add);                                                      \
      (add)->next = NULL;                                                        \
  } else {                                                                       \
      (head)=(add);                                                              \
      (head)->prev = (head);                                                     \
      (head)->next = NULL;                                                       \
  }                                                                              \
} while (0);

#define DL_DELETE(head,del)                                                      \
do {                                                                             \
  if ((del)->prev == (del)) {                                                    \
      (head)=NULL;                                                               \
  } else if ((del)==(head)) {                                                    \
      (del)->next->prev = (del)->prev;                                           \
      (head) = (del)->next;                                                      \
  } else {                                                                       \
      (del)->prev->next = (del)->next;                                           \
      if ((del)->next) {                                                         \
          (del)->next->prev = (del)->prev;                                       \
      } else {                                                                   \
          (head)->prev = (del)->prev;                                            \
      }                                                                          \
  }                                                                              \
} while (0);


#define DL_FOREACH(head,el)                                                      \
    for(el=head;el;el=el->next)

/******************************************************************************
 * circular doubly linked list macros                                         *
 *****************************************************************************/
#define CDL_PREPEND(head,add)                                                    \
do {                                                                             \
 if (head) {                                                                     \
   (add)->prev = (head)->prev;                                                   \
   (add)->next = (head);                                                         \
   (head)->prev = (add);                                                         \
   (add)->prev->next = (add);                                                    \
 } else {                                                                        \
   (add)->prev = (add);                                                          \
   (add)->next = (add);                                                          \
 }                                                                               \
(head)=(add);                                                                    \
} while (0)

#define CDL_DELETE(head,del)                                                     \
do {                                                                             \
  if ( ((head)==(del)) && ((head)->next == (head))) {                            \
      (head) = 0L;                                                               \
  } else {                                                                       \
     (del)->next->prev = (del)->prev;                                            \
     (del)->prev->next = (del)->next;                                            \
     if ((del) == (head)) (head)=(del)->next;                                    \
  }                                                                              \
} while (0);

#define CDL_FOREACH(head,el)                                                     \
    for(el=head;el;el= (el->next==head ? 0L : el->next)) 


#endif /* UTLIST_H */

