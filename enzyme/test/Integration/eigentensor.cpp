// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -mllvm -force-vector-width=1 -ffast-math -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -S | %lli - 
// O0 not supported as need type information
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O3 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O2 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O1 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | %lli - 
// RUN: %clang++ -I/usr/include/eigen3 -Xclang -new-struct-path-tbaa -fno-unroll-loops -fno-vectorize -fno-slp-vectorize -fno-exceptions -O0 %s -S -emit-llvm -o - | %opt - %loadEnzyme -enzyme -enzyme_nonmarkedglobals_inactive=1 -enzyme_inline=1 -S | %lli - 

#define EIGEN_NO_AUTOMATIC_RESIZING 1
#define EIGEN_DONT_ALIGN 1
#define EIGEN_NO_DEBUG 1
#define EIGEN_UNROLLING_LIMIT 0
#define EIGEN_DONT_VECTORIZE 1

#include "test_utils.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Tensor;

constexpr size_t IN = 4, OUT = 4, NUM = 5;

/*
namespace Eigen {
template<typename Derived>
struct TensorEvaluator<Derived, DefaultDevice>
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, DefaultDevice>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions > 0 ?
                               internal::traits<Derived>::NumDimensions : 0;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = (internal::unpacket_traits<PacketReturnType>::size > 1),
    Layout = Derived::Layout,
    CoordAccess = NumCoords > 0,
    RawAccess = true
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const DefaultDevice& device)
      : m_data(const_cast<typename internal::traits<Derived>::template MakePointer<Scalar>::Type>(m.data())), m_dims(m.dimensions()), m_device(device), m_impl(m)
  { }

  // Used for accessor extraction in SYCL Managed TensorMap:
  const Derived& derived() const { return m_impl; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* dest) {
    if (dest) {
      for(unsigned i=0, len=m_dims.TotalSize(); i<len; i++) {
        dest[i] = m_data[i];
      }
      //m_device.memcpy((void*)dest, m_data, sizeof(Scalar) * m_dims.TotalSize());
      return false;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return m_data[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) {
    eigen_assert(m_data);
    return m_data[index];
  }

  template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt<PacketReturnType, LoadMode>(m_data + index);
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    return internal::pstoret<Scalar, PacketReturnType, StoreMode>(m_data + index, x);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<DenseIndex, NumCoords>& coords) const {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(const array<DenseIndex, NumCoords>& coords) {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized,
                        internal::unpacket_traits<PacketReturnType>::size);
  }

  EIGEN_DEVICE_FUNC typename internal::traits<Derived>::template MakePointer<Scalar>::Type data() const { return m_data; }

  /// required by sycl in order to construct sycl buffer from raw pointer
  const DefaultDevice& device() const{return m_device;}

 protected:
  typename internal::traits<Derived>::template MakePointer<Scalar>::Type m_data;
  Dimensions m_dims;
  const DefaultDevice& m_device;
  const Derived& m_impl;
};


template<typename Derived>
struct TensorEvaluator<const Derived, DefaultDevice>
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, DefaultDevice>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions > 0 ?
                               internal::traits<Derived>::NumDimensions : 0;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = (internal::unpacket_traits<PacketReturnType>::size > 1),
    Layout = Derived::Layout,
    CoordAccess = NumCoords > 0,
    RawAccess = true
  };

  // Used for accessor extraction in SYCL Managed TensorMap:
  const Derived& derived() const { return m_impl; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const DefaultDevice& device)
      : m_data(m.data()), m_dims(m.dimensions()), m_device(device), m_impl(m)
  { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* data) {
    if (!NumTraits<typename internal::remove_const<Scalar>::type>::RequireInitialization && data) {
      //m_device.memcpy((void*)data, m_data, m_dims.TotalSize() * sizeof(Scalar));
      for(unsigned i=0, len=m_dims.TotalSize(); i<len; i++) {
        data[i] = m_data[i];
      }
      return false;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return loadConstant(m_data+index);
  }

  template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt_ro<PacketReturnType, LoadMode>(m_data + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<DenseIndex, NumCoords>& coords) const {
    eigen_assert(m_data);
    const Index index = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? m_dims.IndexOfColMajor(coords)
                        : m_dims.IndexOfRowMajor(coords);
    return loadConstant(m_data+index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized,
                        internal::unpacket_traits<PacketReturnType>::size);
  }

  EIGEN_DEVICE_FUNC typename internal::traits<Derived>::template MakePointer<const Scalar>::Type data() const { return m_data; }

  /// added for sycl in order to construct the buffer from the sycl device
  const DefaultDevice& device() const{return m_device;}

 protected:
  typename internal::traits<Derived>::template MakePointer<const Scalar>::Type m_data;
  Dimensions m_dims;
  const DefaultDevice& m_device;
  const Derived& m_impl;
};

template<typename LeftArgType, typename RightArgType>
struct TensorEvaluator<const TensorAssignOp<LeftArgType, RightArgType>, DefaultDevice>
{
  typedef TensorAssignOp<LeftArgType, RightArgType> XprType;
  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, DefaultDevice>::type PacketReturnType;
  typedef typename TensorEvaluator<RightArgType, DefaultDevice>::Dimensions Dimensions;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;

  enum {
    IsAligned = TensorEvaluator<LeftArgType, DefaultDevice>::IsAligned & TensorEvaluator<RightArgType, DefaultDevice>::IsAligned,
    PacketAccess = TensorEvaluator<LeftArgType, DefaultDevice>::PacketAccess & TensorEvaluator<RightArgType, DefaultDevice>::PacketAccess,
    Layout = TensorEvaluator<LeftArgType, DefaultDevice>::Layout,
    RawAccess = TensorEvaluator<LeftArgType, DefaultDevice>::RawAccess
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const DefaultDevice& device) :
      m_leftImpl(op.lhsExpression(), device),
      m_rightImpl(op.rhsExpression(), device)
  {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<LeftArgType, DefaultDevice>::Layout) == static_cast<int>(TensorEvaluator<RightArgType, DefaultDevice>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
  }

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // The dimensions of the lhs and the rhs tensors should be equal to prevent
    // overflows and ensure the result is fully initialized.
    // TODO: use left impl instead if right impl dimensions are known at compile time.
    return m_rightImpl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar*) {
    eigen_assert(dimensions_match(m_leftImpl.dimensions(), m_rightImpl.dimensions()));
    m_leftImpl.evalSubExprsIfNeeded(NULL);
    // If the lhs provides raw access to its storage area (i.e. if m_leftImpl.data() returns a non
    // null value), attempt to evaluate the rhs expression in place. Returns true iff in place
    // evaluation isn't supported and the caller still needs to manually assign the values generated
    // by the rhs to the lhs.
    return m_rightImpl.evalSubExprsIfNeeded(m_leftImpl.data());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_leftImpl.cleanup();
    m_rightImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalScalar(Index i) {
    m_leftImpl.coeffRef(i) = m_rightImpl.coeff(i);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void evalPacket(Index i) {
    const int LhsStoreMode = TensorEvaluator<LeftArgType, DefaultDevice>::IsAligned ? Aligned : Unaligned;
    const int RhsLoadMode = TensorEvaluator<RightArgType, DefaultDevice>::IsAligned ? Aligned : Unaligned;
    m_leftImpl.template writePacket<LhsStoreMode>(i, m_rightImpl.template packet<RhsLoadMode>(i));
  }
  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_leftImpl.coeff(index);
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const
  {
    return m_leftImpl.template packet<LoadMode>(index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    // We assume that evalPacket or evalScalar is called to perform the
    // assignment and account for the cost of the write here, but reduce left
    // cost by one load because we are using m_leftImpl.coeffRef.
    TensorOpCost left = m_leftImpl.costPerCoeff(vectorized);
    return m_rightImpl.costPerCoeff(vectorized) +
           TensorOpCost(
               numext::maxi(0.0, left.bytes_loaded() - sizeof(CoeffReturnType)),
               left.bytes_stored(), left.compute_cycles()) +
           TensorOpCost(0, sizeof(CoeffReturnType), 0, vectorized, PacketSize);
  }

  /// required by sycl in order to extract the accessor
  const TensorEvaluator<LeftArgType, DefaultDevice>& left_impl() const { return m_leftImpl; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<RightArgType, DefaultDevice>& right_impl() const { return m_rightImpl; }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return m_leftImpl.data(); }

 private:
  TensorEvaluator<LeftArgType, DefaultDevice> m_leftImpl;
  TensorEvaluator<RightArgType, DefaultDevice> m_rightImpl;
};



};
*/

extern "C" {
    extern int diffe_const;
    extern double __enzyme_autodiff(void*, const Tensor<float, 2>* __restrict K, const Tensor<float, 2>* __restrict Kp, const Tensor<float, 4>* __restrict I, const Tensor<float, 4>* __restrict Ip, Tensor<float, 4>* __restrict O, Tensor<float, 4>* __restrict Op);
}

__attribute__((noinline))
static void matvec(const Tensor<float, 2>* __restrict K, const Tensor<float, 4>* __restrict In, Tensor<float, 4>* Out) {
  Eigen::array<ptrdiff_t, 2> dims({1, 2});
  *Out = In->convolve(*K, dims);
}

int main(int argc, char** argv) {

    Tensor<float, 4> input(3, 3, 7, 11);
    Tensor<float, 2> kernel(2, 2);
    Tensor<float, 4> output(3, 2, 6, 11);
    input.setRandom();
    kernel.setRandom();

    Tensor<float, 4> inputp(3, 3, 7, 11);
    Tensor<float, 2> kernelp(2, 2);
    Tensor<float, 4> outputp(3, 2, 6, 11);
    inputp.setZero();
    kernelp.setZero();
    outputp.setRandom(); //One();

    matvec(&kernel, &input, &output);
    printf("did original\n");
    __enzyme_autodiff((void*)matvec, &kernel, &kernelp, &input, &inputp, &output, &outputp);
    Tensor<float, 2> expected_kernel(2, 2);
for (int i = 0; i < 3; ++i) {
  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 6; ++k) {
      for (int l = 0; l < 11; ++l) {
        const float result = output(i,j,k,l);
        const float expected = input(i,j+0,k+0,l) * kernel(0,0) +
                               input(i,j+1,k+0,l) * kernel(1,0) +
                               input(i,j+0,k+1,l) * kernel(0,1) +
                               input(i,j+1,k+1,l) * kernel(1,1);
        //VERIFY_IS_APPROX(result, expected);
        //VERIFY_IS_APPROX(result, expected);
		for(int si=0; si<2; si++)
		for(int sj=0; sj<2; si++)
			expected_kernel(si,sj) += outputp(i, j, k, l) * input(i, j+si, k+sj, l);
      }
    }
  }
}
 

	for(int si=0; si<2; si++)
	for(int sj=0; sj<2; si++) {
        fprintf(stderr, "kernelp(si=%d, sj=%d)=%f\n", si, sj, kernelp(si, sj));
        APPROX_EQ( kernelp(si, sj), expected_kernel(si, sj), 1e-10);
    }
     
}
